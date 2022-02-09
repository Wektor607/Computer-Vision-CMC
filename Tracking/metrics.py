
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max((xB - xA, 0)) * max((yB - yA), 0)
    
    if interArea == 0:
        return 0

    boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """
    
    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {obj[0]: obj[1:] for obj in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for id in matches.keys():
            if((id in obj_dict.keys()) and (matches[id] in hyp_dict.keys()) and (iou_score(obj_dict[id], hyp_dict[matches[id]]) > threshold)):
                dist_sum += iou_score(obj_dict[id], hyp_dict[matches[id]])
                match_count += 1
                del obj_dict[id]
                del hyp_dict[matches[id]]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        pair = []
        for ido in obj_dict.keys():
            for idh in hyp_dict.keys():
                if(iou_score(obj_dict[ido], hyp_dict[idh]) > threshold):
                    pair.append([ido, idh, iou_score(obj_dict[ido], hyp_dict[idh])])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        pair.sort(key=lambda x: -x[2])
        for pair in pair:
            if((not pair[0] in matches.keys()) and (not pair[1] in matches.values())):
                dist_sum += pair[2]
                match_count += 1
                matches[pair[0]] = pair[1]
        # Step 5: Update matches with current matched IDs

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    n = 0
    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {obj[0]: obj[1:] for obj in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        n += len(obj_dict)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for idx_obj, idx_hyp in matches.items():
            if((idx_obj in obj_dict) and (idx_hyp in hyp_dict)):
                dist = iou_score(obj_dict[idx_obj], hyp_dict[idx_hyp])
                if(dist > threshold):
                    dist_sum += dist
                    match_count += 1
                    del obj_dict[idx_obj]
                    del hyp_dict[idx_hyp]
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        iou = []
        for idx_obj, bbox_obj in obj_dict.items():
            for idx_hyp, bbox_hyp in hyp_dict.items():
                metric = iou_score(bbox_obj, bbox_hyp)
                if(metric > threshold):
                    iou.append((idx_obj, idx_hyp, metric))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections 
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        # Step 6: Update matches with current matched IDs
        used_obj = set()
        used_hyp = set()
        for idx_obj, idx_hyp, metric in iou:
            if((idx_obj not in used_obj) and (idx_hyp not in used_hyp)):
                used_obj.add(idx_obj)
                used_hyp.add(idx_hyp)
                if((idx_obj in matches) and (matches[idx_obj] != idx_hyp)):
                    mismatch_error += 1
                matches[idx_obj] = idx_hyp
                dist_sum += metric
                match_count += 1
                del obj_dict[idx_obj]
                del hyp_dict[idx_hyp]
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(hyp_dict)
        missed_count += len(obj_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / n

    return MOTP, MOTA
