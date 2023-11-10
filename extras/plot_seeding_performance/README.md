# Seeding performance plotter

This Python script plots the performance of seeding code if the
`nseed_performance_writer` class is used to analyse it. The script takes a file
of seeds and a file of tracks (with default arguments) and optional filter and
output flags. For example, simply calling the script should work fine in most
situations (if the CSV files are in the current directory):

```sh
$ python extras/plot_seeding_performance/plot.py
```

Input and output files can be specified explicitly:

```sh
$ python extras/plot_seeding_performance/plot.py \
    --seeds seeds.csv \
    --tracks tracks.csv \
    --output plot.pdf
```

Filtering ranges can also be specified:

```sh
$ python extras/plot_seeding_performance/plot.py --eta=-2.7:2.7:51 --phi=::10 --pt=1::51
```

This will plot the eta range of -2.7 to 2.7 in 51 bins, autodetermine the phi
range over 10 bins, and plot over a pT range between 1 and an autodetermined
maximum in 51 bins.
