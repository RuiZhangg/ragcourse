## Ragcourse ![](https://github.com/RuiZhangg/ragcourse/workflows/tests/badge.svg)
This project performs rag for answering questions regarding courses and course registration. Database for Harvey Mudd Collge courses is provided.

### Overview

`ragcourse` contains the python scripts for performing rag search, evaluating the rag for Harvey Mudd course information, and building databases.

`mudd_cs.db` and `mudd.db` are databases using FTS5 for Computer Science major at Harvey Mudd College and for all majors at Harvey Mudd College respectively.

`mudd_cs_course` and `mudd_course` are benchmarks for testing the rag search with `mudd_cs.db` database and `mudd.db` database respectvely. The benchmarks containing true or false statements with answers to test if the rag answers correctly.


### Usage

To use this rag search, get your GROQ_API_KEY and add it to your environment, then run
```
$ python3 ragcourse/__init__.py
```
For example,
```
$ python3 ragcourse/__init__.py
ragcourse> What are prerequisites for CSCI070?
According to ARTICLE0, the prerequisites for CSCI070 HM - Data Structures and Program Development are:

1. At least one of the following: CSCI060 HM or CSCI042 HM
2. At least one mathematics course at the level of calculus or higher. Additionally, it is recommended that you take MATH055 HM.
```

You can provide web urls to build your own database, such as
```
$ python3 ragcourse/__init__.py --url=https://www.cmc.edu/academic/departments-majors-programs --db=cmc.db
```

There is also test for evaluating the quality of the RAG system provided.

To use the evaluation, run 
```{bash}
$ python3 ragcourse/evaluate.py
Question: I can take 10 credits and be a full time student
Actual labels: False
Predicted labels: False
----------------------------------------------------------------------
......
----------------------------------------------------------------------
Success: 65
Failure: 9
Success ratio: 0.88
```

Currently, it constantly gets a success ratio larger than 0.85 on `mudd_course` benchmark evaluation with `mudd.db` database.
