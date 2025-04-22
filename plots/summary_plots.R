library(readr)
library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)
library(knitr)
library(kableExtra)
library(lubridate)

paragraph_turns <- read_csv("../data/paragraph_turns_with_emotions.csv")
episodes_all <- read_csv("../data/episodes_with_id.csv")
episodes <- read_csv("../data/filtered_episodes_with_id.csv")

names(paragraph_turns)
names(episodes)

################################################################################################
# EPISODES
################################################################################################

# host
host_counts <- episodes %>% count(host, sort = TRUE)
host_counts_filtered <- host_counts %>%
  filter(n >= 10)
kable(host_counts_filtered, format = "latex", booktabs = TRUE, caption = "Unique Hosts with Counts") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

# podTitle
pod_counts <- episodes %>% count(podTitle, sort = TRUE)
pod_counts_filtered <- pod_counts %>%
  filter(n >= 5)
kable(pod_counts_filtered, format = "latex", booktabs = TRUE, caption = "Unique Podcast Titles with Counts") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

# category1
category_counts <- episodes %>% count(category1, sort = TRUE)
kable(category_counts, format = "latex", booktabs = TRUE, caption = "Unique Podcast Categories with Counts") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

# date
episodes <- episodes %>% mutate(createdOn = as.Date(episodeDateLocalized))
episodes <- episodes %>% mutate(date = ymd_hms(episodeDateLocalized))
episodes <- episodes %>% mutate(date = as.Date(date))
episodes_by_day <- episodes %>% count(date)
pdf("episodes_time_series.pdf", width = 8, height = 6)
highlight_dates <- as.Date(c("2020-05-25", "2020-06-03"))
highlight_points <- episodes_by_day[episodes_by_day$date %in% highlight_dates, ]
ggplot(episodes_by_day, aes(x = date, y = n)) +
  geom_line() +
  geom_point(data = highlight_points, aes(x = date, y = n), 
             color = "red", size = 3) +  # customize dot color/size here
  labs(x = "Date",
       y = "Number of Episodes") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16)
  )
dev.off()

# date for selected vs all episodes
episodes <- episodes %>% mutate(createdOn = as.Date(episodeDateLocalized))
episodes <- episodes %>% mutate(date = ymd_hms(episodeDateLocalized))
episodes <- episodes %>% mutate(date = as.Date(date))
episodes_by_day <- episodes %>% count(date)
episodes_all <- episodes_all %>% mutate(createdOn = as.Date(episodeDateLocalized))
episodes_all <- episodes_all %>% mutate(date = ymd_hms(episodeDateLocalized))
episodes_all <- episodes_all %>% mutate(date = as.Date(date))
episodes_all_by_day <- episodes_all %>% count(date)
episodes_by_day$source <- "Episodes"
episodes_all_by_day$source <- "Episodes_all"
combined_data <- bind_rows(episodes_by_day, episodes_all_by_day)
ggplot(combined_data, aes(x = date, y = n, color = source)) +
  geom_line() +
  labs(x = "Date", y = "Number of Episodes") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16)
  )

# duration
episodes$durationMinutes <- episodes$durationSeconds / 60
pdf("episodes_duration.pdf", width = 8, height = 6)
episodes_binned <- episodes %>%
  mutate(bin = cut(durationMinutes,
                   breaks = seq(0, 140, by = 10),
                   right = FALSE, include.lowest = TRUE)) %>%
  filter(!is.na(bin)) %>%
  droplevels() %>%
  count(bin)
episodes_binned$bin <- gsub("\\[|\\)", "", episodes_binned$bin)  # Remove square and parentheses
episodes_binned$bin <- gsub(",", "-", episodes_binned$bin)  # Replace comma with hyphen
episodes_binned$bin <- factor(episodes_binned$bin, levels = unique(episodes_binned$bin))
ggplot(episodes_binned, aes(x = bin, y = n)) +
  geom_col(fill = "black") +
  geom_text(aes(label = n), hjust = -0.2) +
  labs(x = "Duration (minutes)", y = "Number of Episodes") +
  coord_flip() +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 14),       # Tick labels
    axis.title = element_text(size = 16)  # Axis titles
  )
dev.off()

# numMainSpeakers
speakers_counts <- episodes %>% count(numMainSpeakers, sort = TRUE)
kable(speakers_counts, format = "latex", booktabs = TRUE, caption = "Unique Speakers with Counts") %>%
  kable_styling(latex_options = c("hold_position", "striped"))

################################################################################################
# PARAGRAPH TURNS
################################################################################################
