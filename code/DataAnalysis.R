### Charts
#setwd("~/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2")

library(dplyr)
library(ggplot2)
library(readr)
library(RColorBrewer)
library(maps)
library(viridis)

new_train <- read_csv("~/VesnaLi/Courses/STAT628_Data_Science_Practicum/Module2/data/new_train.csv")

# Wrong longitude & latitude
new_train %>% filter(longitude > 20, longitude < 120) %>%
         select(city, longitude, latitude) %>% 
  group_by(longitude, latitude, city) %>% summarize(count = n())

new_train %>% filter(city == 'henderson') %>% group_by(longitude, latitude) %>% summarize(count = n())





### Time Series

reviews_monthly_stars <- new_train %>% group_by(date=substr(date,1,7), stars) %>% summarize(count=n()) %>%
  arrange(desc(date))
reviews_monthly_stars <- droplevels(reviews_monthly_stars)

# Time Series Stacked
#title=paste("# of Yelp Reviews by Month, by # Stars")
p1 <- ggplot(aes(x = as.POSIXct(paste(date,"-01",sep="")), y=count, fill=as.factor(stars)), data=reviews_monthly_stars) +
  geom_area(position = "stack", alpha = .8) +
#  scale_x_datetime(breaks = date_breaks("1 year"), labels = date_format("%Y")) +
#  scale_y_continuous(label = comma) +
#  theme_custom() + 
  theme(legend.title = element_blank(), legend.position="bottom",
        legend.direction="horizontal", legend.key.width=unit(0.25, "cm"),
        legend.key.height=unit(0.25, "cm")) +
  labs(x="Date of Review Submission",
       y="Total # of Review Submissions (by Month)") +
  theme(panel.grid =element_blank()) +
  scale_fill_wsj()
#  scale_fill_brewer(palette="YlGn",
#                    labels = c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"))
#  scale_fill_manual(values = c("#c49a1d", "#ffffcc", "#92d4a5", "#37a8b7", "#246aa9"),
#                    labels = c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"))
#  scale_color_manual(values = c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"),
#                     labels = c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"))

p1
# Time Series Density

p2 <- ggplot(aes(x=as.POSIXct(paste(date,"-01",sep="")), y=count, fill=as.factor(stars)), data=reviews_monthly_stars) +
  geom_area(position = "fill") +
  theme(legend.title = element_blank(), legend.position="bottom",
        legend.direction="horizontal", legend.key.width=unit(0.25, "cm"),
        legend.key.height=unit(0.25, "cm")) +
  labs(x="Date of Review",
       y="Proportion of All Yelp Reviews") +
  theme(panel.grid =element_blank()) +
  scale_fill_manual(values = c("#c49a1d", "#ffffcc", "#92d4a5", "#37a8b7", "#246aa9"),
                    labels = c("1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"))

p2

### Length of reviews
nword_stars <- new_train %>% filter(nword <= 1000) %>%
  group_by(nword) %>%
  summarize(rate = mean(stars), count = n(), variance = var(stars)) %>%
  arrange(nword)

p3 <- ggplot(aes(x = nword, y = rate, color = variance, size = 1/count) , data = nword_stars) +
  geom_point() +
  scale_color_gradientn(colours = terrain.colors(3)) +
  labs(x="Length of Review",
       y="Average rating") +
  theme(panel.grid = element_blank()) + 
  theme(legend.position="none")
p3


len_stars <- new_train %>% group_by(length) %>% summarize(rate = mean(stars), count = n()) %>% arrange(length)

p4 <- ggplot(aes(x = length, y = rate, color = count) , data = len_stars) +
  geom_point(alpha = .8) +
  scale_color_gradientn(colours = terrain.colors(10)) +
  labs(x="Length of Review",
       y="Average rating") +
  theme(panel.grid = element_blank()) +
  theme(legend.position="none")
p4


### Map
city_stars <- new_train %>% group_by(city) %>%
  summarize(rate = mean(stars), count = n(), lati = mean(latitude), longi = mean(longitude), c_count = n()) %>%
  group_by(lati,longi) %>%
  arrange(desc(count))

library(ggalt)
library(ggthemes)

for(i in 1:length(city_stars$city)){
  if(city_stars$count[i] > 100000){
    city_stars$c_count[i] = 5
  } else if(city_stars$count[i] > 10000){
    city_stars$c_count[i] = 4
  } else if(city_stars$count[i] > 100){
    city_stars$c_count[i] = 3
  } else if(city_stars$count[i] > 10){
    city_stars$c_count[i] = 2
  } else{
    city_stars$c_count[i] = 1
  }
}

world <- map_data("world")
world <- world[world$region != "Antarctica",] # Remove Antarctica
world <- world %>% filter(lat > 20, long < 50)

p5 <- ggplot() + geom_map(data=world, map=world, aes(long, lat, map_id = region),
                color="white", fill="#7f7f7f", size=0.05, alpha=1/4)

p5 <- p5 + geom_point(data = city_stars, 
                      aes(x = longi, y = lati, color=rate, size = as.factor(c_count)), 
                      shape=20, alpha = 0.4) +
  scale_color_gradient(low = "black",high = "orange") +
  scale_size_manual(values = c(5:10)) +
  theme(panel.grid = element_blank()) +
  labs(x="",
       y="") 
#  theme(legend.position="none")
p5


### Upper words
ncapital_stars <- new_train %>%
  group_by(ncapital) %>%
  summarize(rate = mean(stars), count = n(), variance = var(stars)) %>%
  arrange(ncapital)

p6 <- ncapital_stars %>% filter(ncapital < 100) %>%
  ggplot(aes(x = ncapital, y = rate, color = variance, size = 1/count)) +
  geom_point() +
  scale_color_gradientn(colours = terrain.colors(3)) +
  labs(x="# of emphatic words",
       y="Average rating") +
  theme(panel.grid = element_blank()) + 
  theme(legend.position="none")
p6

p7 <- new_train %>% group_by(ncapital, stars) %>%
  summarize(count = n()) %>%
  filter(ncapital <= 10) %>%
  ggplot(aes(x=ncapital, y=count, fill=as.factor(stars))) +
  geom_bar(stat="identity", position=position_dodge())
p7

count_stars <- new_train %>% group_by(stars) %>% summarize(count = n())
excla_stars <- new_train %>% group_by(excla, stars) %>%
  summarize(count = n())
for(i in 1:294){
  excla_stars$count[i] <-
    excla_stars$count[i]/as.numeric(count_stars$count[count_stars$stars == excla_stars$stars[i]])
}


p8 <- excla_stars %>%
  filter(excla < 5) %>%
  ggplot(aes(x=excla, y=count, fill=as.factor(stars))) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_fill_wsj() +
  theme(panel.grid = element_blank()) + 
  theme(legend.position="none")
p8


p9 <- excla_stars %>% filter(excla < 10) %>%
  ggplot(aes(x = "", y = count, fill = as.factor(excla))) + 
  geom_bar(stat = "identity", width = 1) +    
  coord_polar(theta = "y") + 
  labs(x = "", y = "", title = "") + 
  theme(axis.ticks = element_blank()) + 
  theme(legend.title = element_blank(), legend.position = "bottom") +
  guides(fill=guide_legend(title='# of Exclamation Marks')) +
  theme(axis.text.x = element_blank()) +
  theme(legend.key.size=unit(0.4,'cm')) +
  scale_fill_brewer(palette="Spectral") +
  facet_wrap( ~ stars)
p9
