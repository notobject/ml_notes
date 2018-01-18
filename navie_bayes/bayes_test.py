import navie_bayes
import feedparser as fp

ny = fp.parse("http://newyork.craigslist.org/search/stp?format=rss")
print(ny['entries'])
