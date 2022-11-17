def get_con_list(events, con_list_type, all_trial_type=None, create_duplicate_faces=False, noise_weight = 1):

    if not all_trial_type:
        # List of all trial_type values
        all_trial_type = list(set().union(
            *[set(events[x].trial_type) for x in events]))
        all_trial_type.sort()


    # contrast types [all_trial_type | boiled_down_1| boiled_down_2 | boiled_down_3 | boiled_down_4 | boiled_down_5 | neuro_based ] 
     # Create list of columns to be used in design matrix
    if con_list_type == 'all_trial_type':
        con_list = {x: 1 for x in all_trial_type}
        nuisance_con = {}

    elif con_list_type == 'all_trial_type_random_weight':
        # this is used to set up random weights across movies
        con_list = { None }
        nuisance_con = {x: noise_weight for x in all_trial_type}

    elif con_list_type == 'boiled_down_1':
        con_list = ['animate',
                    {'biological_motion': 1, 'body_parts': 1},
                    {'biological': 1, 'social': 1},
                    'faces',
                    {'inanimate_small': 1, 'tools': 1},
                    'inanimate_big',
                    'non_social',  
                    {'salient_near_away': 1, 'far': 1},
                    'salient_near_towards',
                    'near',
                    { 'closed': 1, 'inside': 1},
                    {'open': 1, 'outside': 1, 'nature': 1},
                    'scene',
                    {'non_biological': 1, 'civilisation': 1},
                    'camera_cut',
                    'scene_change',
                    ]

    elif con_list_type == 'boiled_down_2':
        con_list = ['animate',
                    {'biological_motion': 1, 'body_parts': 1},
                    {'biological': 1, 'social': 1, 'non_social':-1},
                    'faces',
                    {'inanimate_small': 1, 'tools': 1},
                    'inanimate_big',
                    {'salient_near_away': 1, 'far': 1},
                    'salient_near_towards',
                    'near',
                    {'open': 1, 'outside': 1, 'nature': 1, 'closed': -1, 'inside': -1},
                    'scene',
                    {'non_biological': 1, 'civilisation': 1},
                    'camera_cut',
                    'scene_change',
                    ]

    elif con_list_type == 'boiled_down_3':
        con_list = [{'body_parts': 1, 'biological': 1, 'faces':1, 'animate':1, 'biological_motion': 1},
                    {'social': 1},
                    {'inanimate_small': 1, 'closed': 1, 'inside': 1 },
                    {'near':1, 'tools': 1, 'civilisation':1, 'non_biological': 1},
                    {'open': 1, 'outside': 1, 'nature': 1, },
                    {'inanimate_big':1, 'far': 1,  'scene':1, 'non_social':1}
        ]
        
    elif con_list_type == 'boiled_down_4':
        # cluster with cutoff of 1.45
        # from this https://github.com/ClionaOD/perceptual_confounds/blob/7be5ea1aa765c5b93103b3f6992db96f6b20f772/model_efficiency_events_movies_con_all_trial_type_clusters.jpg

        con_list = [{'non_social':1},
                    {'inanimate_big':1, 'far':1},
                    {'scene':1, 'near':1},
                    {'outside':1, 'open':1},
                    {'nature':1},
                    {'tools':1, 'inanimate_small':1},
                    {'non_biological':1, 'civilisation':1},
                    {'inside':1, 'closed':1},
                    {'faces':1, 'animate':1, 'social': 1},
                    {'body_parts':1, 'biological_motion': 1, 'biological':1},
        ]
    elif con_list_type == 'boiled_down_5':
        # (1) problem with boiled down 4 is that power is poor for some, as they are anti-correlated
        # so fix this up
        # (2) also balance contrast weighting better so columns with more regressors summed are not automatically bigger
        # make sum of positive=1 and sum of negative = -1
        # (3) still some negative strong negative correlations, so also made nature vs. non-bio/civ 
        # (4) changed weighting of non_social as getting far too much influence on bio stuff
        con_list = [
                    {'inanimate_big':0.5, 'far':0.5},
                    {'scene':0.5, 'near':0.5},
                    {'outside':0.5, 'open':0.5, 'inside':-0.5, 'closed':-0.5},
                    {'nature':1,  'non_biological':-0.25, 'civilisation':-0.25, 'tools':-0.25, 'inanimate_small':-0.25},
                    {'faces':1/6, 'animate':1/6, 'social': 1/6, 'non_social':-1/6},
                    {'body_parts':1/6, 'biological_motion': 1/6, 'biological':1/6},
        ]            
    elif con_list_type == 'neuro_based':
        # Contrasts explicitly based on the semantic dimensions of
        # Kanwisher/Epstein/Culham/Downing , Konkle , Huth , 
        # Kravitz for PPA being spaces (near/far etc.) rather than semantic factors
        #'outside',  'inside' 
        con_list = ['faces', 'body_parts', 'scene', 'tools',
                    {'animate':1 , 'inanimate_small':-0.5, 'inanimate_big':-0.5},
                    'biological_motion',
                    {'nature':1, 'civilisation':-1},
                    {'social':1 , 'non_social':-1},
                    {'biological':1 , 'non_biological':-1},
                    {'far':1/3, 'open':1/3, 'near':-1/3, 'closed':-1/3 }
        ]
    
    elif con_list_type == 'boiled_down_new':
        # Defined manually from results of clustering on longlist only
        # non-bio/civilisation and tools clusters are anticorrelated to nature/outsisde so they have been grouped
        
        con_list = [
                    {'tools':1/6,'tool_use':1/6,'inanimate_small':1/6,'inside':1/6, 'non_biological':1/6, 'civilisation':1/6, 'nature':-1/2,'outside':-1/2},
                    {'body_parts':1/4, 'biological_motion': 1/4, 'animate':1/4, 'faces':1/4},
                    'social',
                    {'near':0.5,'camera_cut':0.5},
                    {'biological':0.5,'closed':0.5},
                    'scene',
                    {'salient_near_away':1/3, 'open':1/3, 'salient_near_towards':1/3},
                    {'far':1/2,'inanimate_big':1/2},
                    {'non_social':1/2,'scene_change':1/2}
        ]

    elif con_list_type == 'boiled_down_new_2':
        # Defined manually from results of clustering on longlist only
        # non-bio/civilisation and tools clusters are anticorrelated to nature/outsisde so they have been grouped
        
        con_list = [
                    {'tools':1/5,'tool_use':1/5,'inanimate_small':1/5,'non_biological':1/5, 'civilisation':1/5, 'nature':-1/5},
                    {'inside':1, 'outside':-1},
                    {'body_parts':1/5, 'biological':1/5,'biological_motion': 1/5, 'animate':1/5, 'faces':1/5},
                    {'social':1, 'non_social':-1},
                    {'closed':1, 'open':-1},
                    'scene',
                    {'far':1/2,'inanimate_big':1/2, 'near':-1/2}
        ]

    if con_list_type.startswith('boiled_down') or con_list_type.startswith('neuro'):
        # These are nuisance columns that will be put into the design matrix, and the simulated
        #  brain signal with random amplitude and into model but aren't of interest
        nuisance_con = {'contrast_sensitivity_function': noise_weight ,
                        'global_contrast_factor': noise_weight , 'rms_difference': noise_weight }

   
    if create_duplicate_faces:
            import numpy as np
            # Create nuisance column nearly duplicate to faces
            for k, v in events.items():
                isface = v[v["trial_type"] == 'faces']
                isface['trial_type'] = 'faces2'
                isface['onset'] = isface['onset'] + \
                    np.random.normal(scale=0.2, size=(len(isface)))
                events[k] = v.append(isface, ignore_index=True)
            # Add this contrast to the nuisance variables
            # if amplitude zero, this column will be only present it design matrix not signal
            nuisance_con['faces2'] = 0

    return con_list, nuisance_con, all_trial_type