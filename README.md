# wiki-bot-detection

#### Homework II - Masters of Data Science Course at UCU

## Mining Massive Databases 2021

Team: Anastasia Holovenko, Rita Marques dos Santos, Khrystyna Skopyk, Antonina Volkotrub.

### Task description

● Consider all the changes done in the wikipedia as stream.
  ● Check here: https://wikitech.wikimedia.org/wiki/RCStream

● Each action is received in json format. 

● Data is full of bots. There is a flag were programmers can 
indicate that an actions has been done by a bot.

● Using this information as ground truth, develop a system able 
to classify users as bot or human.

● Constrain: You need to sample, and just use the 20% of the 
data stream. 

● Finally, train a Bloom Filter that filter out bots from the stream. 
  ● Find the correct parameters for the bloom filter having an 
error below 10%.

#### Extra points

● Make your system to work with Spark Streaming

● Describe the distribution of edits per users and bots.
