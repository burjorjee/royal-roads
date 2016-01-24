Experiments with a new royal road function (Contingent Parities) that support the [Generative Fixation](http://www.cs.brandeis.edu/~kekib/dissertation.html)
theory of adaptation and the [Hypomixability](http://s3.amazonaws.com/burjorjee/.../efficient_hypomixability_elimination.pdf)
theory of sex. 

A companion blog post ([When will recombinative evolution outperform local search](http://evorithmics.org)) introduces 
the Contingent Parities Function and compares the behavior of recombinative evolution and simulated annealing.

Simulated annealing and a genetic algorithm can be run on a contingent parities function as follows:
 
#### Simulated Annealing:

```python
import royalroads as rr
rr.simulatedAnnealing(rngSeed=1)
```

#### Genetic Algorithm:

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