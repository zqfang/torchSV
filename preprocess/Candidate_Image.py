import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageDraw

from kmeans import kmeans
from Init_Image import pipeup_column, get_rgb

#statistic depth
def get_depth(sam_file, chr_id, pos_l, pos_r):
    read_depth = sam_file.count_coverage(chr_id, pos_l, pos_r)
    depth = np.array(list(read_depth)).sum(axis=0)   
    depth = list(depth)
    return depth

# statistic feature numbers
def get_clip_num(sam_file,chr_id, pos_l, pos_r):
    """get map_type number for each base in the given region.

       return: list of tuples (base_pos, map_type_num)
               if region have not reads, return [].
       see map_type number here,  cigartuples, see here: https://pysam.readthedocs.io/en/latest/api.html
        If the alignment is not present, None is returned.

        The operations are:    
            M	BAM_CMATCH	0
            I	BAM_CINS	1
            D	BAM_CDEL	2
            N	BAM_CREF_SKIP	3
            S	BAM_CSOFT_CLIP	4
            H	BAM_CHARD_CLIP	5
            P	BAM_CPAD	6
            =	BAM_CEQUAL	7
            X	BAM_CDIFF	8
            B	BAM_CBACK	9
    """
    clip_temp = []
    for read in sam_file.fetch(chr_id, pos_l, pos_r):
        # iter over each read(record) in the sam file
        if None != read.cigarstring:
            base_pos = read.get_reference_positions(True) # return all positions each base for a read 
            #read_len = len(read.get_reference_positions(True))
            read_len = len(base_pos)
            index = 0
            for read_map_pos in range(read_len):
                if base_pos[read_map_pos]!=None:
                    break
                else:
                    index += 1

             # in most case, index == 0 , so ...
            read_start = read.reference_start-index
            read_end = read_start+read_len-1

            for i in range(pos_r-pos_l+1):
                if pos_l+i<=read_end and pos_l+i >= read_start:
                    index_ptr=0
                    map_type = -1
                    for cigar in read.cigartuples:
                        # cigartuples, see here: https://pysam.readthedocs.io/en/latest/api.html 
                        # the alignment is returned as a list of tuples of (operation, length)
                        # If the alignment is not present, None is returned.
                        if(pos_l+i>index_ptr):
                            index_ptr = cigar[1]+index_ptr
                            map_type=cigar[0] # map_type: [0,9]
                    clip_temp.append((pos_l+i,-map_type)) # TODO: why negative ?
    if len(clip_temp) < 1: 
        return clip_temp # the cases that regions did not have reads 
    df = pd.DataFrame(clip_temp) 
    clip_record_df = df.groupby(0).sum() # sum records grouby base_pos
    clip_record_df = clip_record_df//4 # TODO: why 4 here => CLIP => split-read value ?
    clip_record = clip_record_df.reset_index().values.tolist()
    return clip_record

# init image
def init_pic(row, col,th,fig, flag):
    if flag=='2d':
        ax = fig.add_subplot(row, col, th)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        return ax
    elif flag == '3d':
        ax = fig.add_subplot(row, col, th, projection='3d')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        return ax



# generate candidate deletion
def call_del(vcf_del, sam_file, del_name, outdir):
    vcf_len = len(vcf_del) # numer of deletions
    print("vcf_len %d"%vcf_len)
    del_pos = []
    for i in range(vcf_len):
        print("i = %d"%i)
        # from Features.py
        # get depths for (chrid, start-200, end+200)
        read_depth = get_depth(sam_file, vcf_del[i][0], int(vcf_del[i][1]-200),int(vcf_del[i][2]+200))
        
        seq_depth = [] # (l_postion, read_depth_per_base)
        len_deletion = len(read_depth)
        for j in range(len_deletion):
            #(l_postion, read_depth_per_base)
            seq_depth.append((int(vcf_del[i][1]-200+j), int(read_depth[j])))

        # # get map_types for (chrid, start-200, end+200)
        ## return (pos, map_type_num)
        clip_pic = get_clip_num(sam_file,vcf_del[i][0], vcf_del[i][1]-200, vcf_del[i][2]+200)

        if len(clip_pic) < 1: continue
        # to numpy array
        seq_depth_array = np.array(seq_depth)
        seq_clip_array = np.array(clip_pic)

        # kmeans for 3 cluster
        result = kmeans(seq_depth_array, 3, 100000)
        result = np.nan_to_num(result)

        class_one = result[result[:,-1]==1,:-1]
        class_two = result[result[:,-1]==2,:-1]
        class_three= result[result[:,-1]==3,:-1]

        # plots
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig = plt.figure()
        ax = init_pic(6,1,1, fig,'2d') # (row, col, th, fig, flag)
        ax.plot(seq_depth_array[:,0], seq_depth_array[:,1], color='r')  
        ax.plot(seq_clip_array[:,0],seq_clip_array[:, 1]*2,color='g')

        ax2 = init_pic(6,1,2,fig,'2d')
        ax2.bar(class_one[:,0],class_one[:,1],color='r')
        ax2.bar(class_two[:,0],class_two[:,1],color='g')
        ax2.bar(class_three[:,0],class_three[:,1],color='b')

        # rolling median to denoise 
        df_depth = pd.DataFrame(seq_depth_array)
        df_clip = pd.DataFrame(seq_clip_array)

        df_merge = pd.merge(df_depth, df_clip, on=[0], how='left') # (pos, depth, map_type_num)
        df_merge_fill = df_merge.fillna(0)
        df_merge_np_roll = np.array(df_merge_fill)

        # uses a 61 bp long sliding window to filter the read depths.
        df_merge_fill_re =  df_merge_fill.iloc[:, 1].rolling(window=61, center=True).median()
        df_merge_fill_re = np.nan_to_num(df_merge_fill_re)
        df_merge_fill_re = df_merge_fill_re[30:-30]*5
        # np.c_ : Stack 1-D arrays as columns into a 2-D array. column_stack
        smooth_kmeans = np.c_[df_merge_np_roll[30:-30, 0], df_merge_fill_re] 
        smooth_kmeans = np.c_[smooth_kmeans, df_merge_np_roll[30:-30, 2]]
        # smooth_kmeans: (pos, depth, map_type_num)

        # cluster
        result_3D = kmeans(smooth_kmeans, 3, 100000)
        result_3D = np.nan_to_num(result_3D)
        ## FIXME: if one, two, tree return empty array, what to do ???
        class_one_3D = result_3D[result_3D[:,-1]==1,:-1]
        class_two_3D = result_3D[result_3D[:,-1]==2,:-1]
        class_three_3D= result_3D[result_3D[:,-1]==3,:-1]

        class_one_3D = class_one_3D[np.lexsort(class_one_3D[:,::-1].T)]
        class_two_3D = class_two_3D[np.lexsort(class_two_3D[:,::-1].T)]
        class_three_3D = class_three_3D[np.lexsort(class_three_3D[:,::-1].T)]

        class_one_3D_mean = np.mean(class_one_3D[:,1])//5 # TODO: what 5 here for ???
        class_two_3D_mean = np.mean(class_two_3D[:,1])//5
        class_three_3D_mean = np.mean(class_three_3D[:,1])//5

        ## FIXME: if one, two, tree return empty array, what to do ??? to nan first
        class_one_3D_min = np.min(class_one_3D[:,1])//5 if class_one_3D.size else np.nan
        class_two_3D_min = np.min(class_two_3D[:,1])//5  if class_two_3D.size else np.nan
        class_three_3D_min = np.min(class_three_3D[:,1])//5  if class_three_3D.size else np.nan

        class_array_min = [class_one_3D_min, class_two_3D_min, class_three_3D_min]

        class_array = [class_one_3D_mean, class_two_3D_mean, class_three_3D_mean]

        class_sort = np.sort(class_array)
        del_left_pos = 0
        del_right_pos = 0
        if class_sort[0] < 4*class_sort[1]//5 and class_sort[0] < 4*class_sort[2]//5:
            class_min_index = np.argmin(class_array)
            class_max_index = np.argmax(class_array)
            class_mid_index = 3-class_max_index-class_min_index
            seq_depth_dict = dict(seq_depth)
        
            if class_min_index == 0 and int(class_one_3D[0,0]) != smooth_kmeans[0,0] and int(class_one_3D[-1,0]) != smooth_kmeans[-1,0]:
                del_left_pos = int(class_one_3D[0, 0])
                del_right_pos = int(class_one_3D[-1, 0])

                while del_left_pos:
                    if seq_depth_dict[del_left_pos] > class_array_min[class_mid_index] or del_left_pos == seq_depth[0][0]:
                        break
                    del_left_pos -= 1
                while del_right_pos:
                    if seq_depth_dict[del_right_pos] > class_array_min[class_mid_index] or del_right_pos == seq_depth[-1][0]:
                        break
                    del_right_pos += 1

                del_left_diff = int(class_one_3D[0,0] - del_left_pos)
                del_right_diff = int(del_right_pos - class_one_3D[-1,0])
                del_pos.append((del_left_pos, del_right_pos, int(class_one_3D[0,0]), int(class_one_3D[-1,0])))

            elif class_min_index == 1 and int(class_two_3D[0,0]) != smooth_kmeans[0,0] and int(class_two_3D[-1,0]) != smooth_kmeans[-1,0]:
                del_left_pos = int(class_two_3D[0,0])
                del_right_pos = int(class_two_3D[-1,0])

                while del_left_pos:
                    if seq_depth_dict[del_left_pos] > class_array_min[class_mid_index] or del_left_pos == seq_depth[0][0]:
                        break
                    del_left_pos -= 1
                while del_right_pos:
                    if seq_depth_dict[del_right_pos] > class_array_min[class_mid_index] or del_right_pos == seq_depth[-1][0]:
                        break
                    del_right_pos += 1

                del_left_diff = int(class_two_3D[0,0] - del_left_pos)
                del_right_diff = int(del_right_pos - class_two_3D[-1,0])
                del_pos.append((del_left_pos, del_right_pos, int(class_two_3D[0,0]), int(class_two_3D[-1,0])))

            elif class_min_index == 2 and int(class_three_3D[0,0]) != smooth_kmeans[0,0] and int(class_three_3D[-1,0]) != smooth_kmeans[-1,0]:
                del_left_pos = int(class_three_3D[0,0])
                del_right_pos = int(class_three_3D[-1,0])

                while del_left_pos:
                    if seq_depth_dict[del_left_pos] > class_array_min[class_mid_index] or del_left_pos == seq_depth[0][0]:
                        break
                    del_left_pos -= 1
                while del_right_pos:
                    if seq_depth_dict[del_right_pos] > class_array_min[class_mid_index] or del_right_pos == seq_depth[-1][0]:
                        break
                    del_right_pos += 1

                del_left_diff = int(class_three_3D[0,0] - del_left_pos)
                del_right_diff = int(del_right_pos - class_three_3D[-1,0])
                del_pos.append((del_left_pos, del_right_pos, int(class_three_3D[0,0]), int(class_three_3D[-1,0])))

            ## FIXME: class_one, two, three are empty arary ???
            ax3 = init_pic(6,1,3,fig,'3d')
            ax3.scatter(class_one_3D[:,0],class_one_3D[:,1],class_one_3D[:,2],color='r')
            ax3.scatter(class_two_3D[:,0],class_two_3D[:,1],class_two_3D[:,2],color='g')
            ax3.scatter(class_three_3D[:,0],class_three_3D[:,1],class_three_3D[:,2],color='b')
            ax3.set_xlabel('X')
            ax3.set_zlabel('Z') 
            ax3.set_ylabel('Y')
            
            ax4 = init_pic(6,1,4,fig,'2d')
            ax4.plot(class_one_3D[:,0],class_one_3D[:,1],color='r')  
            ax4.plot(class_two_3D[:,0],class_two_3D[:,1],color='g')  
            ax4.plot(class_three_3D[:,0],class_three_3D[:,1],color='b')  
            
            print("left_pos %d"%del_left_pos)
            print("right_pos %d"%del_right_pos)
            if del_right_pos - del_left_pos != 0  and del_right_pos + del_left_pos != 0:

                del_len = del_right_pos - del_left_pos + 1
                del_pic = []
                for del_i in range(del_len):
                    del_pic.append((del_left_pos+del_i, seq_depth_dict[del_left_pos+del_i]))

                del_pic = np.array(del_pic)
                ax5 = init_pic(6,1,5,fig,'2d')

                ax5.plot(del_pic[:,0],del_pic[:,1],color='r')  

        fig.set_size_inches(18.5, 20.5)
        print("last_i= %d"%i)
        fig.savefig(os.path.join(outdir,  f"{del_name}_{i}_{del_left_pos}_{del_right_pos}.png"))
        plt.close('all')
    return del_pos

# divide image
def draw_pic(clip_dict_record, pile_record, del_pos_np_start, deletion_length, outdir):
    blank = Image.new("RGB",[256, 256],"white")

    pile_record_len = len(pile_record)
    drawObject = ImageDraw.Draw(blank)
    y_start_index = 0
    old_x_start = 5
    for j in range(pile_record_len):
            print("-- %d"%pile_record[j][0])
            print("--- %d"%del_pos_np_start)
            x_start = (pile_record[j][0] - del_pos_np_start)*5 + 5
            print("x_start %d "%x_start)
            if old_x_start == x_start:
                old_x_start = x_start
                print("x_pic_start %d"%x_start)
                y_start = 5 + y_start_index*5
                print("y_pic_start %d"%y_start)
                print("y_index %d"%y_start_index)
                y_start_index += 1
                x_end = x_start + 5
                y_end = y_start + 5
                if pile_record[j][0] in clip_dict_record:
                    base_rgb = get_rgb(-clip_dict_record[pile_record[j][0]],pile_record[j])
                else:
                    base_rgb = get_rgb(0,pile_record[j])
                print("rgb")
                print(base_rgb)
                drawObject.rectangle((x_start,y_start, x_end, y_end),fill=base_rgb)
            elif old_x_start != x_start:
                old_x_start = x_start
                print("x_pic_start %d"%x_start)
                y_start_index = 0
                y_start = 5 + y_start_index*5
                print("y_pic_start %d"%y_start)
                y_start_index += 1
                x_end = x_start + 5
                y_end = y_start + 5

                if pile_record[j][0] in clip_dict_record:
                    base_rgb = get_rgb(-clip_dict_record[pile_record[j][0]],pile_record[j])
                else:
                    base_rgb = get_rgb(0,pile_record[j])
                print("rgb")
                print(base_rgb)
                drawObject.rectangle((x_start,y_start, x_end, y_end),fill=base_rgb)

    #../NA20845/true_gene_pic/gene_pic/del_chr2_50_200
    blank.save(os.path.join(outdir, f"del_1_{del_pos_np_start}_{del_pos_np_start + deletion_length}.png"),"PNG")

# get position
def gene_point_pic(chr_id, pos_l, pos_r):
    gene_pic = []
    every_len = pos_l
    while every_len < pos_r:
        gene_tuple = (chr_id, every_len, every_len+50)
        gene_pic.append(gene_tuple)
        every_len = every_len + 50
    if every_len >= pos_r:
        gene_tuple = (chr_id, every_len-50, pos_r)
        gene_pic.append(gene_tuple)
    return gene_pic