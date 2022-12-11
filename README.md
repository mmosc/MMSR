# MMSR

## Report
Link to report: https://www.overleaf.com/read/hjvjcvffndyf


## FastAPI Introduction
install FastAPI on conda environment:
```conda install -c conda-forge fastapi```

start the Webserver:
```uvicorn main:app --reload```

on standard settings the Webserver can be reached on:
```http://127.0.0.1:8000/docs```

### Possible Requests 
```/query/?artist=ARTIST&track=TRACK&top=NVALUE&vectorData=VECTOR&simFunction=SIMFUNC```

The query parameters values should be specified as **ARTIST** the name of the artist,  **TRACK** the song of the artist, **NVALUE** the number of songs to return as similar.

**VECTOR** could have 3 values *tfidf*, *word2vec* or *bert*

**SIMFUNC** could be *cosineSim*, *innerProduct* or *jaccardSim*


```/metrics/?k=KVALUE&vectorData=VECTORDATA&simFunction=SIMFUNCTION```

The query parameters values shoud be specified as **KVALUE** to get the @k metrics

**VECTORDATA** could have 3 values *tfidf*, *word2vec* or *bert*

**SIMFUNCTION** could be *cosineSim*, *innerProduct* or *jaccardSim*

## Extra Data:
### Task 1
Top 100 Song id's : https://1drv.ms/u/s!AgdGFQd2-hyCiQPMTqJA4OvQ5glM?e=f8OV5W

### Task 2
Top 100 Song id's : https://1drv.ms/u/s!AgdGFQd2-hyCiQQ1B3FPm0nBGWrZ?e=4wiE9S
