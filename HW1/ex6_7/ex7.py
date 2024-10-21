import os

file_name = "beers.txt"

os.system(f"curl -o {file_name} -su datamining2021:Data-Min1ng-2021 http://aris.me/contents/teaching/data-mining-2024/protected/beers.txt")
beers = {}

with open(file_name) as file:
    for line in file:
        line = line.strip().split("\t")
        beer = line[0]
        mark = int(line[1])
        if beer not in beers:
            beers[beer] = [mark, 1]
        else:
            beers[beer][0] += mark
            beers[beer][1] += 1

os.remove(file_name)

#filter out beers with <100 reviews
for beer in list(beers.keys()):
    if beers[beer][1] < 100:
        beers.pop(beer)
    else:
        total = beers[beer][0]
        count = beers[beer][1]
        beers[beer] = total/count

#sort the beers by their average review mark (descending)
beers_by_avg_score = sorted(beers.keys(), key=lambda beer: beers[beer], reverse=True)

print("Top 10 Beers with at least 100 reviews:")
for beer in beers_by_avg_score[:10]: 
    print(f"{beer}: {beers[beer]}")

