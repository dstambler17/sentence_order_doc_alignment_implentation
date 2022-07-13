import argparse
import re

from utils.common import regex_extractor_helper, replace_newline

IDX_RE = re.compile(r'\[(.*?)\]', flags=re.IGNORECASE)


def build_aligned_sents(idxs: str, text_list: list):
    '''
    Builds full sentence based on vecalign produced idx alignments
    '''
    final_sent = ''
    for s_idx in idxs.split(','):
        s_idx = int(s_idx)
        try:
            sent = text_list[s_idx]
        except Exception as e:
            import pdb
            pdb.set_trace()
        if len(final_sent) == 0:
            final_sent += sent
        else:
            final_sent += ' '
            final_sent += sent
    return final_sent+ '\n'

def print_out_sents(save_path, lang, sents):
    final_path = '%s-%s.alignments' % (save_path , lang)    
    with open(final_path, 'w') as f:
        f.writelines(sents)


def read_in_alignments(src_doc_path, tgt_doc_path, indx_paths, threshold):
    path_root = '/'.join(src_doc_path.split('/')[:-1])
    
    src_text_list, tgt_text_list = None, None
    with open(src_doc_path, 'r') as f:
        src_text_list = f.read().split('\n')
    with open(tgt_doc_path, 'r') as f:
        tgt_text_list = f.read().split('\n')
        
    if not src_text_list or not tgt_text_list:
        raise ValueError("Lists can't be none")
    
    final_src_sents, final_tgt_sents = [], []

    with open(indx_paths, 'r') as f:
        for line in f:
            src_idx_raw, tgt_idx_raw, score_raw = line.split(':')

            src_idxs = regex_extractor_helper(IDX_RE, src_idx_raw)
            tgt_idxs = regex_extractor_helper(IDX_RE, tgt_idx_raw)
            score = float(replace_newline(score_raw))

            if score > threshold and src_idxs and tgt_idxs: 
                src_sent = build_aligned_sents(src_idxs, src_text_list)
                tgt_sent = build_aligned_sents(tgt_idxs, tgt_text_list)
                
                final_src_sents.append(src_sent)
                final_tgt_sents.append(tgt_sent)
    
    return final_src_sents, final_tgt_sents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_doc_path' , help='path to the file with source docs', required=True)
    parser.add_argument('--tgt_doc_path', help='Dataset folder that contains all data. Ex: wmt16/processed', required=True)
    parser.add_argument('--index_path' , help='path to the file with indicies', required=True)
    parser.add_argument('-t', '--threshold', help='documents with lower match score will be skipped', default=0.00, type=float, required=False)
    parser.add_argument('-e', '--src_lang', help='src langs', default='en', type=str, required=False)
    parser.add_argument('-f', '--target_lang', help='target langs', default='fr', type=str, required=False)
    parser.add_argument('-o', '--output_path', help='Output file base bad', required=True)

    args = parser.parse_args()

    src_sents, tgt_sents = read_in_alignments(args.src_doc_path, args.tgt_doc_path, args.index_path, args.threshold)

    
    print_out_sents(args.output_path, args.src_lang, src_sents)
    print_out_sents(args.output_path, args.target_lang, tgt_sents)



