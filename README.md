## Ragcourse ![](https://github.com/RuiZhangg/ragcourse/workflows/tests/badge.svg)
This project performs rag for answering questions regarding courses and course registration. Database for Harvey Mudd Collge courses is provided.

To use this rag search, get your GROQ_API_KEY and add it to your environment, then run
```
python3 ragcourse/__init__.py
```
For example,
```
python3 ragcourse/__init__.py
ragcourse> What are prerequisites for CSCI070?
According to ARTICLE0, the prerequisites for CSCI070 HM - Data Structures and Program Development are:

1. At least one of the following: CSCI060 HM or CSCI042 HM
2. At least one mathematics course at the level of calculus or higher. Additionally, it is recommended that you take MATH055 HM.
```

You can provide web urls to build your own database, such as
```
python3 ragcourse/__init__.py --url=https://www.cmc.edu/academic/departments-majors-programs --db=cmc.db
```

There is also test for evaluating the quality of the RAG system provided.

To use this file, also clone the submodule hairy-trumpt, get your GROQ_API_KEY and add it to your environment, then run 
```{bash}
python3 ragcourse/evaluate.py
```
The result can be checked in test.

Currently, it constantly gets a success ratio larger than 0.85 on `mudd_course` benchmark evaluation.
