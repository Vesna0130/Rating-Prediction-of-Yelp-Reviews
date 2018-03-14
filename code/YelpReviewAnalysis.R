#setwd("~/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2")

library(caret)
library(tm)
library(wordcloud)
library(xtable)
library(readr)
library(dplyr)
library(tidyverse)

rank_colors = c(brewer.pal(9, "Reds")[c(8,7,6)],brewer.pal(9, "Greens")[c(7,8)])


### Wordclouds
# 1-gram
dict <- read_csv("data/words_by_stars.csv")

for(i in 1:5){
  png(filename = paste0("plot/Yelp_Words_", i, "star.png"), width = 3000, height = 3000, res= 300)
  subdict = dict[which(dict$star == i), ]
  wordcloud(words = subdict$words, freq = subdict$freq, min.freq = 1,
             max.words = 300, random.order = FALSE, rot.per = 0.35, 
             colors = brewer.pal(8, "Dark2"))
  dev.off()
}

# 2-gram
dict <- read_csv("data/words_by_stars_2gram.csv")

for(i in 1:5){
  png(filename = paste0("plot/Yelp_Words_", i, "star_2gram.png"), width = 3000, height = 3000, res= 300)
  subdict = dict[which(dict$star == i), ]
  wordcloud(words = subdict$words, freq = subdict$freq, min.freq = 1,
            max.words = 300, random.order = FALSE, rot.per = 0.35, 
            colors = brewer.pal(8, "Dark2"))
  dev.off()
}