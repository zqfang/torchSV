import pysam

class FILE:
    def get_bed_file(self, bed_path, isdel):
        deletion_pos = open(bed_path,'r')
        if deletion_pos == None:
            print('bed is empty')
            return
        bed_detail = []
        bed_upAnchor = []
        bed_downAnchor = []

        if isdel == 'deletion': # 3 column file: chr, start, end
            for temp in deletion_pos:
                tmp = temp.strip().split('\t')
                chr_id = tmp[0]
                scan_l_pos = int(tmp[1]) 
                scan_r_pos = int(tmp[2])
                bed_tuples = (chr_id, scan_l_pos, scan_r_pos)
                bed_detail.append(bed_tuples)
            return bed_detail

        elif isdel == 'non_deletion_upAnchor':
            for temp in deletion_pos:
                tmp = temp.split('\t')
                chr_id = tmp[0]
                scan_l_pos = int(tmp[1])
                scan_r_pos = int(tmp[2])
                print("scan_l_pos %d"%scan_l_pos)
                print("scan_r_pos %d"%scan_r_pos)
                del_length = scan_r_pos - scan_l_pos + 1
                if del_length >700:
                    del_length = 4 * del_length // 5
                upAnchor_l_pos = scan_l_pos - del_length - 150
                upAnchor_r_pos = upAnchor_l_pos + del_length
                print("upAnchor_l_pos %d"%upAnchor_l_pos)
                print("upAnchor_r_pos %d"%upAnchor_r_pos)
                bed_tuples = (chr_id, upAnchor_l_pos, upAnchor_r_pos)
                bed_upAnchor.append(bed_tuples)
            return bed_upAnchor

        elif isdel == 'non_deletion_downAnchor':
            for temp in deletion_pos:
                tmp = temp.split('\t')
                chr_id = tmp[0]
                scan_l_pos = int(tmp[1])
                scan_r_pos = int(tmp[2])
                print("scan_l_pos %d"%scan_l_pos)
                print("scan_r_pos %d"%scan_r_pos)
                del_length = scan_r_pos - scan_l_pos + 1
                if del_length >700:
                    del_length = 4 * del_length // 5
                downAnchor_l_pos = scan_r_pos + 150
                downAnchor_r_pos = downAnchor_l_pos + del_length
                print("upAnchor_l_pos %d"%downAnchor_l_pos)
                print("upAnchor_r_pos %d"%downAnchor_r_pos)
                bed_tuples = (chr_id, downAnchor_l_pos, downAnchor_r_pos)
                bed_downAnchor.append(bed_tuples)
            return bed_downAnchor

    @staticmethod
    def get_sam_file(bam_path):
        sam_file = pysam.AlignmentFile(bam_path,"rb")
        if sam_file == None:
            print("bam_file is empty")
            return
        return sam_file

class BED(FILE):
    def __init__(self, bed_file):
        if isinstance(bed_file, str):
            self.bed = self.get_bed_file(bed_file,'deletion')
        else:
            self.bed = None # list of tuples (chr, start, end)
        self.bed_file = bed_file
        self.del_50_200 = []
        self.del_200_700 = []
        self.del_700_1000 = []
        self.del_1000 = []
        self.del_other = []
        self.bed = None

    def bed_class(self):
        """split records into different intervals"""  
        if self.bed is None:
            self.bed = self.get_bed_file(self.bed_file,'deletion')
            
        for bed in self.bed:
            if(abs(bed[1]-bed[2])>=50 and abs(bed[1]-bed[2])<200):
                self.del_50_200.append(bed)
            elif(abs(bed[1]-bed[2])>=200 and abs(bed[1]-bed[2])<700):
                self.del_200_700.append(bed)
            elif(abs(bed[1]-bed[2])>=700 and abs(bed[1]-bed[2])<1000):
                self.del_700_1000.append(bed)
            elif(abs(bed[1]-bed[2])>=1000):
                self.del_1000.append(bed)
            else:
                self.del_other.append(bed)

class BAM(FILE):
    def __init__(self):
        self.read_len = 0  
        self.read_cnt = 0
        self.bam_record = []
        self.pile_record = []
        self.clip_record = []
        self.depth = []

    def read_record(self, sam_file, chr_id, pos_l, pos_r):
        for read in sam_file.fetch(chr_id, pos_l, pos_r):
            if None != read.cigarstring:
                self.bam_record.append(read)
                print(read.reference_start)
                print(read)
            self.read_cnt = len(self.bam_record)
            self.read_len = self.bam_record[self.read_cnt-1].infer_read_length()