
import argparse
import sys

from utils.common import get_doc_filename_from_url, load_extracted
from Levenshtein import distance

JOIN_TOKEN = "JOIN_TOKEN"
EDIT_DIST_THRESH = 0.05

#TODO: Change this later to be more flexible
DIR_BASE_PATH = '/home/dstambl2/doc_alignment_implementations/data/wmt16_dev/processed/'


def load_doamin_true_pairs(recall_urls_file_path: str, domain_names: list):
    domain_lines = []
    with open(recall_urls_file_path) as f:
        for line in f:
            line_name = JOIN_TOKEN.join(line.split())
            if not domain_names: # If empty, assume, all domain_names
                 domain_lines.append(line_name)
            else:
                for dom in domain_names:
                    if dom in line_name:
                        domain_lines.append(line_name)
    return domain_lines
        

    
def load_aligned_cand_pairs(aligned_pair_doc_file_count):
    matching_domains = []
    tracker = 0
    with open(aligned_pair_doc_file_count) as f:
        for line in f:
            tracker += 1
            line_arr = line.split('\t')
            doc_one, doc_two = line_arr[-2], line_arr[-1]
            matching_domains.append(doc_one + JOIN_TOKEN + doc_two)
    print(tracker)
    return matching_domains


def get_src_tgt_dict(pairs):
    src_tgt_dict = {}
    for pair in pairs:
        src, tgt = pair.split(JOIN_TOKEN)
        src_tgt_dict[src] = tgt
    return src_tgt_dict


def load_doc(url, lang):
    file_name = get_doc_filename_from_url(url, lang)
    path = DIR_BASE_PATH + file_name
    print(path)
    
    docs = load_extracted(path)
    return docs[url]


def compute_edit_distance(tgt_cand_url, true_tgt_url, lang):
    print(tgt_cand_url)
    print(true_tgt_url)
    tgt_cand_doc = load_doc(tgt_cand_url, lang)
    true_tgt_doc = load_doc(true_tgt_url, lang)
    
    levin_dist = distance(tgt_cand_doc, true_tgt_doc) #levenshtein_distance(tgt_cand_doc, true_tgt_doc) 
    
    print(levin_dist, tgt_cand_url, true_tgt_url)
    print(100* levin_dist / float(max(len(tgt_cand_doc), len(true_tgt_doc))))

    return  100* levin_dist / float(max(len(tgt_cand_doc), len(true_tgt_doc)))


def get_correctly_aligned_pairs(candidate_pairs, all_possible_pairs, true_src_tgt_dict, lang):
    '''
    See if candidates are in the true pair list
    '''
    aligned_pairs = []
    pair_dict = {pair: True for pair in all_possible_pairs}
    for cand in candidate_pairs:
        cand_clean = cand.replace('\n', '')
        source_cand, tgt_cand = cand_clean.split(JOIN_TOKEN)[0], cand_clean.split(JOIN_TOKEN)[1]
        if cand_clean in pair_dict:
            aligned_pairs.append(cand)
        elif source_cand in true_src_tgt_dict: #TODO: Change to elif once done debug
            true_tgt = true_src_tgt_dict[source_cand]
            if compute_edit_distance(tgt_cand, true_tgt, lang) < EDIT_DIST_THRESH:
                aligned_pairs.append(cand)

    
    return aligned_pairs


def compute_recall_score(all_possible_pairs, extracted_pairs):
    all_possible_pairs_count = len(all_possible_pairs)
    extracted_pairs_count = len(extracted_pairs)
    recall_score = extracted_pairs_count / all_possible_pairs_count
    print("""
        All Possible Pair Count: %s,
        Extracted Pair Count: %s,
        Recall Score: %.3f
    """ % (all_possible_pairs_count, extracted_pairs_count, recall_score))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aligned_pair_doc', help='aligned pairs document', required=True)
    parser.add_argument('--all_aligned_urls', help='Document with all aligned urls', required=True)

    parser.add_argument('-d', '--domains', nargs='+', default=[])
    parser.add_argument('-l', '--target_lang', help='language of target doc', required=True)

    args = parser.parse_args()

    algo_generated_pairs = load_aligned_cand_pairs(args.aligned_pair_doc)
    true_pairs = load_doamin_true_pairs(args.all_aligned_urls, args.domains)
    true_src_tgt_dict = get_src_tgt_dict(true_pairs)

    aligned_true_pairs = get_correctly_aligned_pairs(algo_generated_pairs, true_pairs, true_src_tgt_dict, args.target_lang)

    compute_recall_score(true_pairs, aligned_true_pairs)



