Dependencies:
	: will run in python3
	: install these libraries- numpy, scipy, sklearn, nltk, matplotlib, pandas
	: also after installing nltk, download stopwords-
		$ python3           (start python3 shell)
		>>> nltk.download('stopwords')
	: then run using python3
	: things to experiment with-
		line> svd_model = TruncatedSVD(n_components=...so on
			change the n_components (number of features documents and queries get reduced to before performing cosine similarity)
		line> max_num_docs = 10

		in latentSI.py
		line> svd_model = TruncatedSVD(...
			try changing n_components (number of topics) and n_iter (for training the model, for accuracy)
