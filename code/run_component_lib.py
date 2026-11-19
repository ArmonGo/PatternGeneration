from Process.render import render_img, from_img_to_url, from_array_to_img
import numpy as np
from PIL import Image
from Process.color_cut import get_color_masked_img, quantifize_color, get_color_set, extract_ink_masks
from Agent.api import api_key
from Process.segmentation import fill_outer_contours, enhance_ink_connection, watershed_segments
from Process.segmentation import project_segments_to_original
from Process.similarity_score import *
from Process.similarity_score import score_candidates
import copy
import os
from functools import partial
import pickle
from Process.clustering import cluster_color_groups, plot_mask_clusters
import pickle
from Process.clustering import build_prototype_library
from Agent.score_agent import recolor_with_judge


API_KEY = api_key(path="../")

def read_file(p, selected_pattern):
    img = render_img(f"../ai_patterns/svg/{selected_pattern}.svg",
                 long_side = 1024, pad = 0)
    pic_img = from_array_to_img(img, save_path = p + f"/Out/ink/{selected_pattern}.png" )
    return pic_img

def color_img(pic_img, p, selected_pattern):
    print("Calling color agent...")
    # call agent to color image
    img_url = from_img_to_url(pic_img)
    print('get url ')
    res = recolor_with_judge(pic_img, API_KEY, p + "/Out/colored/", selected_pattern)
    # first mask the original sections 
    colored_img = Image.open(p + f"/Out/colored/{selected_pattern}.png")
    mask_img = Image.open(p + f"/Out/ink/{selected_pattern}.png")
    out_path = p + f"/Out/selected/{selected_pattern}.png"
    selected_img = get_color_masked_img(colored_img, mask_img, out_path,
                                         return_img=True, if_img=False)
    # quantifize the color 
    rst, quantized_rgb = quantifize_color(selected_img)
    Image.fromarray(quantized_rgb).save(p + f"/Out/quantized/{selected_pattern}.png")
    # save the quantizing results 
    with open(p + f"/Out/quantized/{selected_pattern}_result.pickle", 'wb') as handle:
            pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def extract_inkmask_per_color(p, selected_pattern):
    # Load images
    quantized_path = p + f"/Out/quantized/{selected_pattern}.png"
    quantized_img = Image.open(quantized_path)
    # extract the each color mask and save them
    colors, counts = get_color_set(quantized_img)
    print(colors)
    out_dir = os.path.join(p, "Out", "inkmask", f"{selected_pattern}")
    os.makedirs(out_dir, exist_ok=True)
    rst = extract_ink_masks(original_path=p + f"/Out/ink/{selected_pattern}.png", 
                  colored_path= p + f"/Out/quantized/{selected_pattern}.png",
                  output_path =  p + f"/Out/inkmask/{selected_pattern}/", 
                  colors = colors, counts = counts, 
                  min_area=20, black_threshold=30)
    metadata = {
        "selected_pattern": selected_pattern,
        "colors": colors,
        "counts": counts, 
        "color_groups" : rst
    }
    with open(p + f"/Out/inkmask/{selected_pattern}/result.pickle", 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata 

def find_the_best_peak_ratio(filled_mask, enhanced):
    scores_list = {}
    for r in np.arange(0.05, 0.6, 0.05):
        r = round(r, 2).item()
        markers, watershed_result = watershed_segments(filled_mask, peak_ratio=r, show=False)
        scores_list[r] ={}
        scores_list[r]['peak_ration'] = r
        candidates_masks = project_segments_to_original(markers, enhanced)
        candidates_list = list(candidates_masks.values())
        s = score_candidates(candidates_list, 
                            lambda_penalty=0.05, alpha=0.8,
                            score_metric="hog"
                            )
        scores_list[r]["score"] = s
        scores_list[r]["candidates_masks"] = candidates_masks
        scores_list[r]["markers"] = markers
    s_ls = {}
    for k, n in scores_list.items():
        s_ls[k] = n["score"]
    best_r = max(s_ls, key=s_ls.get)
    return best_r, scores_list

def split_each_color_mask(meta_data, p):
    data_save = copy.deepcopy(meta_data)
    selected_pattern = meta_data['selected_pattern']
    color_groups = meta_data['color_groups']
    new_colors = []
    for single_color in color_groups:
        raw_info = copy.deepcopy(single_color)
        mask_p = single_color['mask_path']
        mask =  np.array(Image.open(mask_p))
        # enhance the image with dilation and closing
        enhanced = enhance_ink_connection(mask, kernel_size=2, iterations=2,
                                   dilation_iteration=1, show=False)
        # fill the mask 
        filled_mask = fill_outer_contours(enhanced, show=False)
        # find the best peak ratio for watershed segmentation
        best_r, scores_list = find_the_best_peak_ratio(filled_mask, enhanced)
        # project the segments to the original mask
        # use enhanced mask to calculate the score
        markers, watershed_result = watershed_segments(filled_mask, peak_ratio=best_r, show=False)
        candidates_masks = project_segments_to_original(markers, enhanced)
        # log the results
        raw_info['best_peak_ratio'] = best_r
        raw_info['scores_list'] = scores_list
        raw_info['candidates_masks'] = candidates_masks
        new_colors.append(raw_info)
    data_save['color_groups'] = new_colors
    out_dir = os.path.join(p, "Out", "components_masks")
    os.makedirs(out_dir, exist_ok=True)
    with open(p + f"/Out/components_masks/{selected_pattern}_result.pickle", 'wb') as handle:
        pickle.dump(data_save, handle,)
    return data_save


def clsutering_candidates(color_groups, distance_metric ='hu_moment', norm= True,
                          show = True, 
                        **metric_kwargs):
    color_id_groups = {}
    for i in color_groups:
        color_id_groups[i['color_id']] = []
        for c, v in i['candidates_masks'].items():
            color_id_groups[i['color_id']].append(v)
    if distance_metric =='hu_moment':
        # masks are already normalised below when norm=True, so skip it inside hu_distance
        fn = partial(hu_distance, norm=not norm)
    elif distance_metric == "chamfer_distance":
        fn = chamfer_distance
    elif distance_metric == "hog":
        fn = None   # pairwise distances come from mask_distance below
    if norm:
        for cid, masks in color_id_groups.items():
            color_id_groups[cid] = [normalize_mask(m) for m in masks] # normalize the masks for clustering
    mask_distance = None
    if distance_metric == "hog":
        # the rotation/mirror search is costly per pair, so compute all masks at once
        all_masks = [m for masks in color_id_groups.values() for m in masks]
        mask_distance, _ = hog_distance_map(all_masks)
    mask_labels, group_labels, group_distance, best_score  = cluster_color_groups(color_id_groups,
                                                                                  distance_fn =fn,
                                                                                  mask_distance=mask_distance)
    if show: 
        plot_mask_clusters(color_id_groups, mask_labels, n_cols=3)
    return mask_labels, group_labels, group_distance, best_score
    
def run_pipe(selected_pattern_list):
    p = "C:/Users/Yameng/Documents/Research/patterngeneration/Manual/"
    for selected_pattern in selected_pattern_list:
        pic_img = read_file(p, selected_pattern)
        # call img
        img_url = from_img_to_url(pic_img)  
        res = color_img(pic_img, p, selected_pattern)
        colored_img = Image.open(p + f"/Out/colored/{selected_pattern}.png")
        meta_data = extract_inkmask_per_color(p, selected_pattern)
        data_colors = split_each_color_mask(meta_data, p)
        # reload it the mask 
        with open(p + f"/Out/components_masks/{selected_pattern}_result.pickle", 'rb') as handle:
            data_save = pickle.load( handle)

        # build prototype
        color_id_groups = {}
        for i in data_save['color_groups']:
            color_id_groups[i['color_id']] = []
            for c, v in i['candidates_masks'].items():
                color_id_groups[i['color_id']].append(v)
        # save 
        mask_labels, group_labels, group_distance, best_score = clsutering_candidates(data_save['color_groups'],
                                                                               distance_metric ='hog',
                                                                               show = False,
                                                                               norm=True)
        lib = build_prototype_library(color_id_groups, mask_labels, distance_metric='hog')
        with open(p + f"/Out/lib/{selected_pattern}_result.pickle", 'wb') as handle:
            pickle.dump( lib, handle)
        print(f'{selected_pattern} done!')
