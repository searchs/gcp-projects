#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Prices using Tensorflow + Cloud ML Engine
# 
# This notebook will show you how to create a tensorflow model, train it on the cloud in a distributed fashion across multiple CPUs or GPUs, explore the results using Tensorboard, and finally deploy the model for online prediction. We will demonstrate this by building a model to predict housing prices.
# 

# In[18]:


import pandas as pd
import tensorflow as tf


# In[19]:


print(tf.__version__)


# ## Tensorflow APIs
# <img src="assets/TFHierarchy.png"  width="50%">
# <sup>(image: https://www.tensorflow.org/images/tensorflow_programming_environment.png)</sup>
# 
# Tensorflow is a heirarchical framework. The further down the heirarchy you go, the more flexibility you have, but that more code you have to write. A best practice is start at the highest level of abstraction. Then if you need additional flexibility for some reason drop down one layer. 
# 
# For this tutorial we will be operating at the highest level of Tensorflow abstraction, using the Estimator API.

# ## Steps
# 
# 1. Load raw data
# 
# 2. Write Tensorflow Code
# 
#  1. Define Feature Columns
#  
#  2. Define Estimator
# 
#  3. Define Input Function
#  
#  4. Define Serving Function
# 
#  5. Define Train and Eval Function
# 
# 3. Package Code
# 
# 4. Train
# 
# 5. Inspect Results
# 
# 6. Deploy Model
# 
# 7. Get Predictions

# ### 1) Load Raw Data
# 
# This is a publically available dataset on housing prices in Boston area suburbs circa 1978. It is hosted in a Google Cloud Storage bucket.
# 
# For datasets too large to fit in memory you would read the data in batches. Tensorflow provides a queueing mechanism for this which is documented [here](https://www.tensorflow.org/programmers_guide/reading_data).
# 
# In our case the dataset is small enough to fit in memory so we will simply read it into a pandas dataframe.

# In[20]:


#downlad data from GCS and store as pandas dataframe 
data_train = pd.read_csv(
  filepath_or_buffer='https://storage.googleapis.com/vijay-public/boston_housing/housing_train.csv',
  names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","MEDV"])

data_test = pd.read_csv(
  filepath_or_buffer='https://storage.googleapis.com/vijay-public/boston_housing/housing_test.csv',
  names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","MEDV"])


# In[21]:


data_train.head()


# #### Column Descriptions:
# 
# 1. CRIM: per capita crime rate by town 
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
# 3. INDUS: proportion of non-retail business acres per town 
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
# 5. NOX: nitric oxides concentration (parts per 10 million) 
# 6. RM: average number of rooms per dwelling 
# 7. AGE: proportion of owner-occupied units built prior to 1940 
# 8. DIS: weighted distances to five Boston employment centres 
# 9. RAD: index of accessibility to radial highways 
# 10. TAX: full-value property-tax rate per $10,000 
# 11. PTRATIO: pupil-teacher ratio by town 
# 12. MEDV: Median value of owner-occupied homes

# ### 2) Write Tensorflow Code

# #### 2.A Define Feature Columns
# 
# Feature columns are your Estimator's data "interface." They tell the estimator in what format they should expect data and how to interpret it (is it one-hot? sparse? dense? continous?).  https://www.tensorflow.org/api_docs/python/tf/feature_column
# 
# 
# 

# In[22]:


FEATURES = ["CRIM", "ZN", "INDUS", "NOX", "RM",
            "AGE", "DIS", "TAX", "PTRATIO"]
LABEL = "MEDV"

feature_cols = [tf.feature_column.numeric_column(k)
                  for k in FEATURES] #list of Feature Columns


# #### 2.B Define Estimator
# 
# An Estimator is what actually implements your training, eval and prediction loops. Every estimator has the following methods:
# 
# - fit() for training
# - eval() for evaluation
# - predict() for prediction
# - export_savedmodel() for writing model state to disk
# 
# Tensorflow has several canned estimator that already implement these methods (DNNClassifier, LogisticClassifier etc..) or you can implement a custom estimator. Instructions on how to implement a custom estimator [here](https://www.tensorflow.org/extend/estimators) and see an example [here](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/timeseries/rnn_cloudmle.ipynb).
# 
# For simplicity we will use a canned estimator. To instantiate an estimator simply pass it what Feature Columns to expect and specify a directory for it to output to.
# 
# Notice we wrap the estimator with a function. This is to allow us to specify the 'output_dir' at runtime, instead of having to hardcode it here

# In[23]:


def generate_estimator(output_dir):
  return tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir=output_dir)


# #### 2.C Define Input Function
# 
# Now that you have an estimator and it knows what type of data to expect and how to intepret, you need to actually pass the data to it! This is the job of the input function. 
# 
# The input function returns a (features, label) tuple
# - features: A python dictionary. Each key is a feature column name and its value is the tensor containing the data for that Feature
# - label: A Tensor containing the label column

# In[24]:


def generate_input_fn(data_set):
    def input_fn():
      features = {k: tf.constant(data_set[k].values) for k in FEATURES}
      labels = tf.constant(data_set[LABEL].values)
      return features, labels
    return input_fn


# #### 2.D Define Serving Input Function
# 
# To predict with the model, we need to define a serving input function which will be used to read inputs from a user at prediction time. 
# 
# Why do we need a separate serving function? Don't we input the same features during training as in serving?
# 
# Yes, but we may be *receiving* data in a different format during serving. The serving input function preforms transormations neccessary to get the data provided at prediction time into the format compatible with the Estimator API.
# 
# returns a (features, inputs) tuple
# - features: A dict of features to be passed to the Estimator
# - inputs: A dictionary of inputs the predictions server should expect from the user

# In[25]:


def serving_input_fn():
  #feature_placeholders are what the caller of the predict() method will have to provide
  feature_placeholders = {
      column.name: tf.placeholder(column.dtype, [None])
      for column in feature_cols
  }
  
  #features are what we actually pass to the estimator
  features = {
    # Inputs are rank 1 so that we can provide scalars to the server
    # but Estimator expects rank 2, so we expand dimension
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
  }
  return tf.estimator.export.ServingInputReceiver(
    features, feature_placeholders
  )


# #### 2.E Define Train and Eval Function
# 
# Finally to train and evaluate we use tf.estimator.train_and_evaluate()
# 
# This function is special because it provides consistent behavior across local and distributed environments.
# 
# Meaning if you run on multiple CPUs or GPUs, it takes care of parrallelizing the computation graph across these devices for you! 
# 
# The tran_and_evaluate() function requires three arguments:
# - estimator: we already defined this earlier
# - train_spec: specifies the training input function
# - eval_spec: specifies the eval input function, and also an 'exporter' which uses our serving_input_fn for serving the model
# 
# **Note running this cell will give an error because we haven't specified an output_dir, we will do that later**

# In[26]:


train_spec = tf.estimator.TrainSpec(
                input_fn=generate_input_fn(data_train),
                max_steps=3000)

exporter = tf.estimator.LatestExporter('Servo', serving_input_fn)

eval_spec=tf.estimator.EvalSpec(
            input_fn=generate_input_fn(data_test),
            steps=1,
            exporters=exporter)

tf.estimator.train_and_evaluate(generate_estimator(output_dir), train_spec, eval_spec)


# ### 3) Package Code
# 
# You've now written all the tensoflow code you need!
# 
# To make it compatible with Cloud ML Engine we'll combine the above tensorflow code into a single python file with two simple changes
# 
# 1. Add some boilerplate code to parse the command line arguments required for gcloud.
# 2. Use the learn_runner.run() function to run the experiment
# 
# We also add an empty \__init__\.py file to the folder. This is just the python convention for identifying modules.

# In[28]:


get_ipython().run_cell_magic('bash', '', 'mkdir trainer\ntouch trainer/__init__.py')


# In[29]:


get_ipython().run_cell_magic('writefile', 'trainer/task.py', '\nimport argparse\nimport pandas as pd\nimport tensorflow as tf\nfrom tensorflow.contrib.learn.python.learn import learn_runner\nfrom tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils\n\nprint(tf.__version__)\ntf.logging.set_verbosity(tf.logging.ERROR)\n\ndata_train = pd.read_csv(\n  filepath_or_buffer=\'https://storage.googleapis.com/vijay-public/boston_housing/housing_train.csv\',\n  names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","MEDV"])\n\ndata_test = pd.read_csv(\n  filepath_or_buffer=\'https://storage.googleapis.com/vijay-public/boston_housing/housing_test.csv\',\n  names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","MEDV"])\n\nFEATURES = ["CRIM", "ZN", "INDUS", "NOX", "RM",\n            "AGE", "DIS", "TAX", "PTRATIO"]\nLABEL = "MEDV"\n\nfeature_cols = [tf.feature_column.numeric_column(k)\n                  for k in FEATURES] #list of Feature Columns\n\ndef generate_estimator(output_dir):\n  return tf.estimator.DNNRegressor(feature_columns=feature_cols,\n                                            hidden_units=[10, 10],\n                                            model_dir=output_dir)\n\ndef generate_input_fn(data_set):\n    def input_fn():\n      features = {k: tf.constant(data_set[k].values) for k in FEATURES}\n      labels = tf.constant(data_set[LABEL].values)\n      return features, labels\n    return input_fn\n\ndef serving_input_fn():\n  #feature_placeholders are what the caller of the predict() method will have to provide\n  feature_placeholders = {\n      column.name: tf.placeholder(column.dtype, [None])\n      for column in feature_cols\n  }\n  \n  #features are what we actually pass to the estimator\n  features = {\n    # Inputs are rank 1 so that we can provide scalars to the server\n    # but Estimator expects rank 2, so we expand dimension\n    key: tf.expand_dims(tensor, -1)\n    for key, tensor in feature_placeholders.items()\n  }\n  return tf.estimator.export.ServingInputReceiver(\n    features, feature_placeholders\n  )\n\ntrain_spec = tf.estimator.TrainSpec(\n                input_fn=generate_input_fn(data_train),\n                max_steps=3000)\n\nexporter = tf.estimator.LatestExporter(\'Servo\', serving_input_fn)\n\neval_spec=tf.estimator.EvalSpec(\n            input_fn=generate_input_fn(data_test),\n            steps=1,\n            exporters=exporter)\n\n######START CLOUD ML ENGINE BOILERPLATE######\nif __name__ == \'__main__\':\n  parser = argparse.ArgumentParser()\n  # Input Arguments\n  parser.add_argument(\n      \'--output_dir\',\n      help=\'GCS location to write checkpoints and export models\',\n      required=True\n  )\n  parser.add_argument(\n        \'--job-dir\',\n        help=\'this model ignores this field, but it is required by gcloud\',\n        default=\'junk\'\n    )\n  args = parser.parse_args()\n  arguments = args.__dict__\n  output_dir = arguments.pop(\'output_dir\')\n######END CLOUD ML ENGINE BOILERPLATE######\n\n  #initiate training job\n  tf.estimator.train_and_evaluate(generate_estimator(output_dir), train_spec, eval_spec)')


# ### 4) Train
# Now that our code is packaged we can invoke it using the gcloud command line tool to run the training. 
# 
# Note: Since our dataset is so small and our model is simple the overhead of provisioning the cluster is longer than the actual training time. Accordingly you'll notice the single VM cloud training takes longer than the local training, and the distributed cloud training takes longer than single VM cloud. For larger datasets and more complex models this will reverse

# #### Set Environment Vars
# We'll create environment variables for our project name GCS Bucket and reference this in future commands.
# 
# If you do not have a GCS bucket, you can create one using [these](https://cloud.google.com/storage/docs/creating-buckets) instructions.

# In[30]:


GCS_BUCKET = 'gs://<PROJECT_ID-bucket>' #CHANGE THIS TO YOUR BUCKET
PROJECT = '<PROJECT_ID>' #CHANGE THIS TO YOUR PROJECT ID
REGION = 'us-central1' #OPTIONALLY CHANGE THIS


# In[31]:


import os
os.environ['GCS_BUCKET'] = GCS_BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION


# #### Run local
# It's a best practice to first run locally on a small dataset to check for errors. Note you can ignore the warnings in this case, as long as there are no errors.

# In[32]:


get_ipython().run_cell_magic('bash', '', "gcloud ai-platform local train \\\n   --module-name=trainer.task \\\n   --package-path=trainer \\\n   -- \\\n   --output_dir='./output'")


# #### Run on cloud (1 cloud ML unit)

# First we specify which GCP project to use.

# In[33]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT')


# Then we specify which GCS bucket to write to and a job name.
# Job names submitted to the ml engine must be project unique, so we append the system date/time. Update the cell below to point to a GCS bucket you own.

# In[34]:


get_ipython().run_cell_magic('bash', '', 'JOBNAME=housing_$(date -u +%y%m%d_%H%M%S)\n\ngcloud ai-platform jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=./trainer \\\n   --job-dir=$GCS_BUCKET/$JOBNAME/ \\\n   --runtime-version 1.4 \\\n   -- \\\n   --output_dir=$GCS_BUCKET/$JOBNAME/output')


# In[36]:


get_ipython().run_cell_magic('bash', '', 'gcloud ai-platform jobs describe housing_191204_125248')


# #### Run on cloud (10 cloud ML units)
# Because we are using the TF Estimators interface, distributed computing just works! The only change we need to make to run in a distributed fashion is to add the [--scale-tier](https://cloud.google.com/ml/pricing#ml_training_units_by_scale_tier) argument. Cloud ML Engine then takes care of distributing the training across devices for you!
# 

# In[37]:


get_ipython().run_cell_magic('bash', '', 'JOBNAME=housing_$(date -u +%y%m%d_%H%M%S)\n\ngcloud ai-platform jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=./trainer \\\n   --job-dir=$GCS_BUCKET/$JOBNAME \\\n   --runtime-version 1.4 \\\n   --scale-tier=STANDARD_1 \\\n   -- \\\n   --output_dir=$GCS_BUCKET/$JOBNAME/output')


# #### Run on cloud GPU (3 cloud ML units)

# Also works with GPUs!
# 
# "BASIC_GPU" corresponds to one Tesla K80 at the time of this writing, hardware subject to change. 1 GPU is charged as 3 cloud ML units.

# In[38]:


get_ipython().run_cell_magic('bash', '', 'JOBNAME=housing_$(date -u +%y%m%d_%H%M%S)\n\ngcloud ai-platform jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=./trainer \\\n   --job-dir=$GCS_BUCKET/$JOBNAME \\\n   --runtime-version 1.4 \\\n   --scale-tier=BASIC_GPU \\\n   -- \\\n   --output_dir=$GCS_BUCKET/$JOBNAME/output')


# #### Run on 8 cloud GPUs (24 cloud ML units)
# To train across multiple GPUs you use a [custom scale tier](https://cloud.google.com/ml/docs/concepts/training-overview#job_configuration_parameters).
# 
# You specify the number and types of machines you want to run on in a config.yaml, then reference that config.yaml via the --config config.yaml command line argument.
# 
# Here I am specifying a master node with machine type complex_model_m_gpu and one worker node of the same type. Each complex_model_m_gpu has 4 GPUs so this job will run on 2x4=8 GPUs total. 
# 
# WARNING: The default project quota is 10 cloud ML units, so unless you have requested a quota increase you will get a quota exceeded error. This command is just for illustrative purposes.

# In[39]:


get_ipython().run_cell_magic('writefile', 'config.yaml', 'trainingInput:\n  scaleTier: CUSTOM\n  masterType: complex_model_m_gpu\n  workerType: complex_model_m_gpu\n  workerCount: 1')


# In[40]:


get_ipython().run_cell_magic('bash', '', 'JOBNAME=housing_$(date -u +%y%m%d_%H%M%S)\n\ngcloud ai-platform jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=./trainer \\\n   --job-dir=$GCS_BUCKET/$JOBNAME \\\n   --runtime-version 1.4 \\\n   --config config.yaml \\\n   -- \\\n   --output_dir=$GCS_BUCKET/$JOBNAME/output')


# ### 5) Inspect Results Using Tensorboard
# 
# Tensorboard is a utility that allows you to visualize your results.
# 
# Expand the 'loss' graph. What is your evaluation loss? This is squared error, so take the square root of it to get the average error in dollars. Does this seem like a reasonable margin of error for predicting a housing price?
# 
# To activate TensorBoard within the JupyterLab UI navigate to **File** - **New Launcher**. Then double-click the 'Tensorboard' icon on the bottom row.
# 
# TensorBoard 1 will appear in the new tab. Navigate through the three tabs to see the active TensorBoard. The 'Graphs' and 'Projector' tabs offer very interesting information including the ability to replay the tests.
# 
# You may close the TensorBoard tab when you are finished exploring.

# ### 6) Deploy Model For Predictions
# 
# Cloud ML Engine has a prediction service that will wrap our tensorflow model with a REST API and allow remote clients to get predictions.
# 
# You can deploy the model from the Google Cloud Console GUI, or you can use the gcloud command line tool. We will use the latter method. Note this will take up to 5 minutes.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="housing_prices"\nMODEL_VERSION="v1"\nMODEL_LOCATION=output/export/Servo/$(ls output/export/Servo | tail -1) \n\n#gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME} #Uncomment to overwrite existing version\n#gcloud ai-platform models delete ${MODEL_NAME} #Uncomment to overwrite existing model\ngcloud ai-platform models create ${MODEL_NAME} --regions $REGION\ngcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --staging-bucket=$GCS_BUCKET')


# ### 7) Get Predictions
# 
# There are two flavors of the ML Engine Prediction Service: Batch and online.
# 
# Online prediction is more appropriate for latency sensitive requests as results are returned quickly and synchronously. 
# 
# Batch prediction is more appropriate for large prediction requests that you only need to run a few times a day.
# 
# The prediction services expects prediction requests in standard JSON format so first we will create a JSON file with a couple of housing records.
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'records.json', '{"CRIM": 0.00632,"ZN": 18.0,"INDUS": 2.31,"NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.0900, "TAX": 296.0, "PTRATIO": 15.3}\n{"CRIM": 0.00332,"ZN": 0.0,"INDUS": 2.31,"NOX": 0.437, "RM": 7.7, "AGE": 40.0, "DIS": 5.0900, "TAX": 250.0, "PTRATIO": 17.3}')


# Now we will pass this file to the prediction service using the gcloud command line tool. Results are returned immediatley!

# In[ ]:


get_ipython().system('gcloud ai-platform predict --model housing_prices --json-instances records.json')


# ### Conclusion
# 
# #### What we covered
# 1. How to use Tensorflow's high level Estimator API
# 2. How to deploy tensorflow code for distributed training in the cloud
# 3. How to evaluate results using TensorBoard
# 4. How deploy the resulting model to the cloud for online prediction
# 
# #### What we didn't cover
# 1. How to leverage larger than memory datasets using Tensorflow's queueing system
# 2. How to create synthetic features from our raw data to aid learning (Feature Engineering)
# 3. How to improve model performance by finding the ideal hyperparameters using Cloud ML Engine's [HyperTune](https://cloud.google.com/ml-engine/docs/how-tos/using-hyperparameter-tuning) feature
# 
# This lab is a great start, but adding in the above concepts is critical in getting your models to production ready quality. These concepts are covered in Google's 1-week on-demand Tensorflow + Cloud ML course: https://www.coursera.org/learn/serverless-machine-learning-gcp
