
import os, sys, argparse
import numpy as np
from os.path import dirname, join, abspath

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from FileIO import BED, FILE
from Init_Image import pipeup_column
from Candidate_Image import call_del, get_clip_num, gene_point_pic, draw_pic


# draw deletion image
def draw_del_pic(bed_del, sam_file, del_name, del_called, del_images):
    """
    bed_del: tuples (chr, start, end)
    sam_file: pysam.AlignmentFile
    del_name: del_50_200, del_20_700, ...
    del_called: output directory for refined deletions
    del_images: output directory for model trainning and testing
    """
    os.makedirs(del_called, exist_ok=True)
    os.makedirs(del_images, exist_ok=True)

    print(bed_del[0][0]) # first record chrid
    chr_id = bed_del[0][0]

    # refined breakpoints
    del_pos = call_del(bed_del, sam_file, del_name, outdir=del_called)
    num_del_pos = len(del_pos) # tuples with 6 elements
    del_pos_np = np.array(del_pos) 

    print("num_del_pos %d"%num_del_pos)
    for i in range(num_del_pos):
        # get map_type number
        clip_record = get_clip_num(sam_file, chr_id, del_pos_np[i,0], del_pos_np[i, 1])
        clip_dict_record = dict(clip_record) # to dict
        # print(clip_dict_record)
        gene_pic = gene_point_pic(chr_id, del_pos_np[i,0], del_pos_np[i,1]) 
        #gene_pic: (chr, pos_l, pos_r) => for image pos_l, pos_r
        for each_pic in gene_pic:
            pile_record = pipeup_column(sam_file, each_pic[0], each_pic[1], each_pic[2])
            deletion_length =  each_pic[2] - each_pic[1]
            draw_pic(clip_dict_record, pile_record, each_pic[1], deletion_length, del_images)

def draw_no_del_pic(bed_del, sam_file, del_name):
    """
    for no_del_region, please input a bed file (chr, start, end) where the region is non-deletion.
    """
    num_no_del_pic = len(bed_del)
    for pos in bed_del:
        clip_record = get_clip_num(sam_file,pos[0],pos[1],pos[2])
        clip_dict_record = dict(clip_record)
        print(clip_dict_record)
        gene_pic = gene_point_pic(pos[0],pos[1],pos[2])
        for each_pic in gene_pic:
            pile_record = pipeup_column(sam_file,each_pic[0],each_pic[1],each_pic[2])
            deletion_length = each_pic[2] - each_pic[1]
            draw_pic(clip_dict_record, pile_record,each_pic[1], deletion_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True, help="Bed file (3 column). <chr start end>")
    parser.add_argument("--bam", required=True, help="Bam file.")
    parser.add_argument("--del_length", choices={ "del_50_200", "del_50_200","del_700_1000", "del_1000"}, 
                        default="del_200_700", )
    parser.add_argument("--del_called", default="del_called", help="outdir for called deletions")
    parser.add_argument("--del_images", default="del_images", help="outdir for images converted from seqeuncing files")
    args = parser.parse_args()
    
    # parse 
    bed = BED(args.bed)
    bed.bed_class()
    # bed_path = "/data/bases/fangzq/20200815_SV/DeepSV/test.bed"
    #bed_file = file.get_bed_file(bed_path, 'deletion')   
    if len(bed.bed) < 1:
        print("bed_file is []")
        return

    sam_file =  FILE.get_sam_file(args.bam)
    # --del_length: del_50_200 del_200_700 del_700_1000 del_1000 ...
    if (args.del_length == "del_50_200"):
        if bed.del_50_200 !=[]:
            draw_del_pic(bed.del_50_200, sam_file, del_name=args.del_length, 
            del_called=args.del_called, del_images=args.del_images)
    elif (args.del_length == "del_200_700"):
        if bed.del_200_700 !=[]:
            draw_del_pic(bed.del_200_700, sam_file, del_name=args.del_length, 
            del_called=args.del_called, del_images=args.del_images)
    elif (args.del_length == "del_700_1000"):
        if bed.del_700_1000 !=[]:
            draw_del_pic(bed.del_700_1000, sam_file, del_name=args.del_length, 
            del_called=args.del_called, del_images=args.del_images)
    elif (args.del_length == "del_1000"):
        if bed.del_1000 !=[]:
            draw_del_pic(bed.del_1000, sam_file, del_name=args.del_length, 
            del_called=args.del_called, del_images=args.del_images)
    
if __name__ == '__main__':
    main()







