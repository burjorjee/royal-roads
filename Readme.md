A royal road function in support of the [generative fixation](http://www.cs.brandeis.edu/~kekib/dissertation.html)
theory of adaptation and the [hypomixability](http://s3.amazonaws.com/burjorjee/.../efficient_hypomixability_elimination.pdf)
theory of sex. 

The royal road function is presented and the behavior of recombinative evolution and simulated annealing is compared in 
a companion blog post: [When will recombinative evolution outperform local search](http://evorithmics.org)

#### To run simulated annealing:

```python
import royalroads as rr
rr.simulatedAnnealing(rngSeed=1)
```

#### To run a genetic algorithm:

```python
import royalroads as rr
rr.geneticAlgorithm(rngSeed=1)
```

#### To compare the performance of the two algorithms on the royal road function:

```python
import royal roads as rr
rr.compareAlgorithms(numRuns=40)
```

# Dependencies #

mmh3, matplotlib, joblib