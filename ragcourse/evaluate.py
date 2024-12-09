'''
This file is for evaluating the quality of our RAG system using the Hairy
Trumpet tool/dataset.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ragcourse import ArticleDB
from ragcourse import rag

class RAGEvaluator:
    def __init__(self, db):
        self.db = db

    def predict(self, question, model='llama-3.1-70b-versatile'):
        '''
        >>> model = RAGEvaluator()
        >>> model.predict('I can take 10 credits and be a full time student')
        False
        >>> model.predict('CSCI131 is offered in both Fall and Spring')
        True
        '''

        db = ArticleDB(self.db)
        textprompt = f'''
        I'm going to provide you a sentence.
        And your job is to tell me it is true or false.

        You should not provide any explanation or other extraneous words.
        Valid values include: [True, False]
        INPUT: I can take CSCI81 without finishing CSCI70
        OUTPUT: False

        INPUT: I could finish Computer Science major without taking Algorithm
        OUTPUT: False

        INPUT: I can take 18 credits without overload.
        OUTPUT: True

        INPUT: {question}
        OUTPUT: '''

        output = rag(textprompt, db, keywords_text=question, model = model)
        result = output.split()

        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='mudd.db')
    parser.add_argument('--path', default="mudd_course")
    parser.add_argument('--model', default='llama3-groq-8b-8192-tool-use-preview')
    args = parser.parse_args()
    
    model = RAGEvaluator(args.db)

    success = 0
    failure = 0

    import json
    with open(args.path) as file:
        data = [json.loads(line) for line in file]

    for i, line in enumerate(data):
        print('Question:', line['question'])
        print('Actual labels:', line['answer'])
        prediction = model.predict(line['question'], model = args.model)
        print('Predicted labels:', prediction[0])
        print('-' * 70)
        if (all(x == y for x, y in zip(line['answer'], prediction[0]))):
            success += 1
        else:
            failure += 1

    # Print the results
    print('Success: %d' % success)
    print('Failure: %d' % failure)

    total = success + failure
    if total > 0:
        success_ratio = success / total
        print('Success ratio: %.2f' % success_ratio)
    else:
        print('No data')