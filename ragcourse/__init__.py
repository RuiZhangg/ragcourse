#!/bin/python3

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import random
import logging
import re
import sqlite3
from sqlite3 import OperationalError

import groq

from groq import Groq
import os


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def run_llm(system, user, model='llama-3.1-70b-versatile', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                "role": "user",
                "content": user,
            }
        ],
        model=model,
        seed=seed,
    )
    return chat_completion.choices[0].message.content


def summarize_text(text, model = "llama-3.1-70b-versatile", seed=None):
    system = 'Summarize the input text below including the contents and details. Pay special attention to school, major, graduation requirements and courses. Must include any information related to major, graduation requirements and courses in detail and the school.'
    return run_llm(system, text, model = model, seed=seed)


def extract_keywords(text, model = "llama-3.1-70b-versatile", seed=None):
    '''
    This is a helper function for RAG.
    Given an input text,
    this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Can I', seed=0)
    'Joe Biden President Trump Democrats Nominee Party'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'immigrants border illegal mexico trump policy'
    '''
    system = "Find the most important keywords for the Text for searching, each keyword should be one to two words long. Separate the keywords with blanks no period; output should include the keywords only without prompt in one line. No plural form, like prerequisites should be prerequisite. If there is course code like CSCI070, must keep it."
    key = run_llm(system, "Text: " + text, model = model, seed=seed)
    if seed != None:
        return key
    else:
        while True:
            listing = key.split(" ")
            if len(listing) < 10 :
                break
            else:
                key = run_llm(system, "Text: " + text, model = model)
        logging.info(f"get keywords: {key}")
        return key

def recursive_summary(text, sizeLimit, model = "llama-3.1-70b-versatile"):
    '''
    Split the whole text to smaller sizes and use LLM to summary to get shorter
    result. Repeating until the whole length is with in the sizeLimit
    '''
    while len(text) > sizeLimit:
        textList = text_split(text, sizeLimit)
        newText = ""
        system = 'Summarize the input text below. Output in English.'
        for paragraph in textList:
            newText += run_llm(system, paragraph, model=model) + "\n\n"
        text = newText
    return text



################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\W+')
    sql_dewhite = rex.sub(' ', sql)
    logging.debug(f'SQL: {sql_dewhite}')


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    We use this decorator on the add_urls method below to ensure that a webcrawl continues even if there are errors.
    '''
    
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(str(e))
    return inner_function

def text_split(text, sizeLimit):
    '''
    Split the input text according to line break, and reorganize the sentences
    into paragraph with in sizeLimit to reduce the number of requests
    >>> text_split("AAB\nDGF", 4)
        ["AAB", "DGF"]
    >>> text_split("AAB\nDGF", 10)
        ["AAB\nDGF"]
    >>> text_split("AAB\nDGF\nKBW", 7)
        ["AAB\nDGF", "KBW"]
    '''
    textList = text.split('\n')
    tempList = [textList[0]]
    wordCounter = len(textList[0])
    for i in range(1, len(textList)):
        length = len(textList[i])
        if wordCounter + length < sizeLimit:
            tempList[-1] += "\n" + textList[i]
            wordCounter += length
        else:
            tempList.append(textList[i])
            wordCounter = length
    return tempList


################################################################################
# Crawl website
################################################################################
def fetch_page_data(url):
    # Send an HTTP GET request to the URL
    try:
        response = requests.get(url)
    except:
        logging.info(f"Failed to request the page: {url}")
        return

    # Check if the request was successful
    if response.status_code != 200:
        logging.info(f"Failed to fetch the page. Status code: {response.status_code}")
        return None
    
    # Parse the page content using BeautifulSoup
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
    except:
        logging.info("beautifulSoup fail to parse "+ url)
        return
        
    # Extract the title
    title = soup.title.string if soup.title else "No Title Found"
    
    # Extract the publish date
    publish_date = "Unknown"
    meta_date = soup.find("meta", {"property": "article:published_time"}) or soup.find("meta", {"name": "publish_date"})
    if meta_date and meta_date.get("content"):
        publish_date = meta_date["content"]
    else:
        # Fallback: Search for date patterns in visible text
        text = soup.get_text()
        date_pattern = r'\b(?:\d{1,2}[-/thstndrd\s]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s,]*\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            publish_date = match.group()
    
    # Extract the main content and preserve line breaks
    main_content = soup.find("main")
    content = main_content.get_text(separator="\n").strip() if main_content else "No Content Found"
    
    # Clean up excessive line breaks
    content = re.sub(r'\n\s*\n+', '\n\n', content)  # Replace multiple line breaks with a single blank line
    
    # Extract all links
    links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True)]
    
    # Capture crawl date
    crawl_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "title": title,
        "publish_date": publish_date,
        "crawl_date": crawl_date,
        "content": content,
        "links": links,
        "url": url
    }

################################################################################
# rag
################################################################################

def rag(text, db, keywords_text=None, model = "llama-3.1-70b-versatile"):
    '''
    This function uses retrieval augmented generation (RAG) to generate an LLM response to the input text.
    The db argument should be an instance of the `ArticleDB` class that contains the relevant documents to use.
    '''
    if keywords_text is None:
        keywords_text = text
    for i in range(3):
        keywords = extract_keywords(keywords_text, model = model)
        articles = db.find_articles(keywords, keywords_text=keywords_text, model = model)
        if len(articles) > 0:
            break
    sources = ""
    for i in range(len(articles)):
        sources += "ARTICLE" + str(i) + " content: " + articles[i]["content"] + "\n"

    system = "You are a college adviser. You will be given several articles and a question. Answer the question based on the articles."

    while (True):
        try:
            user = sources + "QUESTION: " + text
            logging.info('rag.SYSTEM: ' + system)
            logging.info('rag.USER' + user)
            anwser = run_llm(system, user, model = model)
            return anwser
        except:
            sources = recursive_summary(sources, len(sources)/2, model = model)

class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    The following example shows how to add urls to the database.

    >>> db = ArticleDB("test.db")
    >>> len(db)
    0
    # >>> db.add_url(ArticleDB._TESTURLS[0])
    # >>> len(db)
    # 1

    Once articles have been added,
    we can search through those articles to find articles about only certain topics.

    >>> articles = db.find_articles('Biology')

    The output is a list of articles that match the search query.
    Each article is represented by a dictionary with a number of fields about the article.

    >>> articles[0]['title']
    Biology Major Program | Biology | Harvey Mudd College
    >>> articles[0].keys()
    ['rowid', 'rank', 'title', 'content', 'url', 'publish_date', 'crawl_date', 'summary', 'depth']
    '''

    _TESTURLS = [
        'https://www.hmc.edu/biology/programs/',
        'https://www.hmc.edu/cs/academic-programs/',
    ]

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory=sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.

        The test below demonstrates that creating a schema on a database that already has the schema will not generate errors.

        >>> db = ArticleDB()
        >>> db._create_schema()
        >>> db._create_schema()
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE articles
            USING FTS5 (
                title,
                content,
                url,
                publish_date,
                crawl_date,
                summary,
                depth
                );
            '''
            self.db.execute(sql)
            self.db.commit()

        # if the database already exists,
        # then do nothing
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed')

    def find_articles(self, query, limit=5, timebias_alpha=1, keywords_text = None, model = "llama-3.1-70b-versatile"):
        '''
        Return a list of articles in the database that match the specified query.

        Lowering the value of the timebias_alpha parameter will result in the time becoming more influential.
        The final ranking is computed by the FTS5 rank * timebias_alpha / (days since article publication + timebias_alpha).
        '''

        sql = '''
        SELECT rowid, title, content, summary, bm25(articles) AS relevance_score
        FROM articles
        WHERE articles MATCH ?
        ORDER BY relevance_score ASC
        LIMIT ?;

        '''

        _logsql(sql)
        cursor = self.db.cursor()
        while True:
            try:
                cursor.execute(sql, (query, limit))
            except OperationalError:
                if keywords_text is not None:
                    query = extract_keywords(keywords_text, model = model)
                continue
            break
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


    @_catch_errors
    def add_url(self, url, model, recursive_depth=0, allow_dupes=False):
        '''
        Download the url, extract various metainformation, and add the metainformation into the db.

        By default, the same url cannot be added into the database multiple times.

        >>> db = ArticleDB("test.db")
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> len(db)
        1

        >>> db.add_url(ArticleDB._TESTURLS[1])
        >>> len(db)
        2

        '''
        if "hmc.edu" not in url:
            logging.info('outside url, skipped:' + url)
            return
        if not allow_dupes:
            sql = '''
            SELECT count(*) FROM articles WHERE url=?;
            '''
            cursor = self.db.cursor()
            cursor.execute(sql, [url])
            row = cursor.fetchone()
            is_dupe = row[0] > 0
            if is_dupe:
                logging.info('With depth '+ str(recursive_depth)+ ' duplicate detected, skip ' + url)
                sql = '''
                SELECT depth FROM articles WHERE url=?;
                '''
                cursor = self.db.cursor()
                cursor.execute(sql, [url])
                depth = cursor.fetchone()                
                if recursive_depth > depth[0]:
                    data = fetch_page_data(url)
                    if data is None:
                        return
                    for link in data['links']:
                        self.add_url(link, model, recursive_depth - 1)
                    record = self.db.execute(
                        "SELECT rowid, * FROM articles WHERE url = ?;", (url,)
                    ).fetchone()
                    updates = {'depth': recursive_depth}
                    set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
                    values = list(updates.values()) + [record['rowid']]
                    update_sql = f"UPDATE articles SET {set_clause} WHERE rowid = ?;"
                    self.db.execute(update_sql, values)
                    self.db.commit()
                return
        data = fetch_page_data(url)
        if data is None:
            return
        logging.info('With depth '+ str(recursive_depth)+ ' inserting into database ' + url)
        if len(data['content']) > 30000:
            logging.info(f"Page to long for summary: {url}")
            if recursive_depth > 0:
                for link in data['links']:
                    self.add_url(link, model, recursive_depth-1)
            return
        try:
            summary = summarize_text(data['content'], model=model)
        except:
            logging.info(f"Fail to summarize for {url}")
            if recursive_depth > 0:
                for link in data['links']:
                    self.add_url(link, model, recursive_depth-1)
            return

        sql = '''
        INSERT INTO articles(title, content, url, publish_date, crawl_date, summary, depth)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        '''
        cursor = self.db.cursor()
        cursor.execute(sql, [
            data['title'],
            data['content'], 
            url,
            data['publish_date'],
            data['crawl_date'],
            summary,
            recursive_depth,
            ])
        self.db.commit()
        logging.info('inserting success')
        if recursive_depth > 0:
            for link in data['links']:
                self.add_url(link, model, recursive_depth-1)
        
    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM articles
        WHERE content IS NOT NULL;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='mudd.db')
    parser.add_argument('--recursive_depth', default=0, type=int)
    parser.add_argument('--url', help='If this parameter is added, then the program will not provide an interactive QA session with the database.  Instead, the provided url will be downloaded and added to the database.')
    parser.add_argument('--model', type=str, required=False, default="llama3-groq-8b-8192-tool-use-preview", help='The model to use for answering or summarizing web pages')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel.upper(),
        )

    db = ArticleDB(args.db)

    if args.url:
        db.add_url(args.url, args.model, recursive_depth=args.recursive_depth, allow_dupes=False)

    else:
        import readline
        while True:
            try:
                text = input('ragcourse> ')
            except EOFError:
                break
            if len(text.strip()) > 0:
                output = rag(text, db, model = args.model)
                print(output)
