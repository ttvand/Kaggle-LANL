# Logic

The logic can broadly be divided into four categories, with the files listed in alphabetical order.


- **Data preparation** - generate or combine analysis data files
	- augment\_train\_test\_gap\_preds.py
	- main\_cpc\_data\_preparation.py
	- save\_gap\_target.py
	- save\_test\_combined.py
	- split\_train\_quake.R
- **Train, validation and test logic** - Main logic for the different modeling approaches
	- train\_valid\_test\_cpc.py
	- train\_valid\_test\_cpc\_main.py
	- train\_valid\_test\_gap.py
	- train\_valid\_test\_lightgbm.py
	- train\_valid\_test\_lightgbm\_sequential.py
	- train\_valid\_test\_rnn.py
	- train\_valid\_test\_rnn\_sequential.py
- **Helper files for the train, validation test scripts**
	- models.py
	- preprocess.py
	- utils.py
- **Additional analysis logic**
	- align\_test\_chunks\_ordered.py
	- analyse\_gap\_preds.R
	- blend\_submissions.py
	- hyperpar\_optim\_lgb.py
	- hyperpar\_optim\_lgb\_sequential.py
	- hyperpar\_optim\_rnn.py
	- hyperpar\_optim\_rnn\_sequential.py
	- hyperpar\_sweep\_gaps.py
	- main\_vae.py
	- post\_deadline\_rescale.py
	- utils\_vae.py
	- verify\_no\_duplicate\_seqs\_test.py
