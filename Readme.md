Experiments with a new royal road function (Contingent Parities) that support the [Generative Fixation](http://www.cs.brandeis.edu/~kekib/dissertation.html)
theory of adaptation and the [Hypomixability](https://s3.amazonaws.com/burjorjee/www/efficient_hypomixability_elimination.pdf)
theory of recombination. 

A companion blog post ([When will evolution outperform local search?](http://blog.evorithmics.org/2016/01/31/when-will-evolution-outperform-local-search/)) introduces 
Contingent Parities Functions and compares the behavior of recombinative evolution and simulated annealing 
on a member of this class of fitness function.

## Usage
Simulated annealing and a genetic algorithm can be run on a contingent parities function of order 2 and height 100 as follows:
 
#### Simulated Annealing:

```python
import royalroads as rr
rr.SA(rngSeed=1, numPeriods=1000)
```

#### Genetic Algorithm:

```python
import royalroads as rr
rr.GA(rngSeed=1, numGenerations=1000, useClamping=False)
```

#### To compare the performance of the two algorithms on the contingent parities function:

```python
import royalroads as rr
rr.compareAlgorithms(numRuns=40)
```

# Dependencies #

```
mmh3, matplotlib, joblib
```