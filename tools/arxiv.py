import argparse

from googletrans import Translator
import arxivpy
import os
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    Title = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_args():
    parser = argparse.ArgumentParser(description='arxiv paper read')
    parser.add_argument(
        '--start', default=0, type=int, help='Start index')
    parser.add_argument(
        '--number', default=5, type=int, help='Number of papers')
    parser.add_argument(
        '--random', default=5, type=int, help='random start index')
    parser.add_argument(
        '--field', default='cv', type=str, help='search field')
    parser.add_argument('--translate', action='store_true', help='Flag of google translate')
    parser.add_argument('--save_file', action='store_true', help='Flag of save file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.system('clear')

    translate = Translator()
    
    if args.random:
        args.start = int(np.random.uniform(low=0, high=10000))
        args.number = int(np.random.uniform(low=1, high=3))
        print("random mode, start: {}, number: {}".format(args.start, args.number))

    if args.field == 'cv':
        search_query = ['cs.CV']
    else:
        search_query = args.field

    print('Searching for {}'.format(search_query))

    articles = arxivpy.query(search_query=search_query,
                            start_index=args.start, max_index=args.start + args.number, results_per_iteration=100,
                            wait_time=5.0, sort_by='lastUpdatedDate') # grab 200 articles
    
    print("Available Keys: ", articles[0].keys())
    # print(articles[1])

    paperlist_file = open("paperlist.txt", "w")

    items = []
    for idx, article in enumerate(articles):
        items.append("============================================ Paper {} ===========================================\n".format(idx+1))
        items.append("Title: \n    {}\n".format(article['title']))
        items.append("Author: \n    {}\n".format(article['authors']))
        items.append("Abstract: \n    {}\n".format(article['abstract']))
        
        if args.translate:
            result = translate.translate(article['abstract'], dest='zh-CN')
            items.append("    翻译: {}\n".format(result.text))
        
        items.append("{}\n".format(article['update_date']))
        items.append("{}\n".format(article['pdf_url']))
    

    for item in items:
        if 'Title' in item:
            print(bcolors.Title + item + bcolors.ENDC)
        else:
            print(item)
        if args.save_file:
            paperlist_file.write(item)

    paperlist_file.close()


