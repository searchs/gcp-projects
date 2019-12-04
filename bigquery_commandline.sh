# BigQuery on Commandline
# Using gCloud interface or SSH into gCloud
gcloud auth list
gcloud config list project

# Check a sample of public data sample
bq show bigquery-public-data:samples.shakespeare
# Check available commands
bq help query
# select a sample from public dataset
bq query --use_legacy_sql=false 'SELECT
   word,
   SUM(word_count) AS count
 FROM
   `bigquery-public-data`.samples.shakespeare
 WHERE
   word LIKE "%raisin%"
 GROUP BY
   word'

# filter public dataset
bq query --use_legacy_sql=false 'SELECT
  word
FROM
  `bigquery-public-data`.samples.shakespeare
WHERE
  word = "huzzah"'

bq ls
# Check other available public dataset
bq ls bigquery-public-data:
# Create a dataset
bq mk babynames
bq ls
# Download raw data§
wget http://www.ssa.gov/OACT/babynames/names.zip
ls
unzip names.zip
ls -al

# table details: datasetID: babynames  tableID: names2010 source: yob2010.txt schema: name:string,gender:string,count:integer
# Load raw data with schema spec
bq load babynames.names2010 yob2010.txt name:string,gender:string,count:integer

bq ls babynames
bq show babynames
bq show babynames.names2010
bq query "SELECT name,count FROM babynames.names2010 WHERE gender = 'F' ORDER BY count DESC LIMIT 5"
bq query "SELECT name,count FROM babynames.names2010 WHERE gender = 'M' ORDER BY count ASC LIMIT 5"
bq rm -r babynames
