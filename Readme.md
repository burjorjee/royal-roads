A royal road function in support of the [generative fixation](http://www.cs.brandeis.edu/~kekib/dissertation.html)
theory of adaptation and the [hypomixability](http://s3.amazonaws.com/burjorjee/.../efficient_hypomixability_elimination.pdf)
theory of sex. 

The behavior of genetic algorithm and simulated annealing is discussed in a companion blog post: 
[When will recombinative evolution outperform local search](http://evorithmics.org)

#### To run simulated annealing:

```python
import royalroads as rr
rr.simulatedAnnealing(1)
```

#### To run a genetic algorithm:

```python
import royalroads as rr
rr.geneticAlgorithm(1)
```

#### To compare the performance of the two algorithms:

```python
import royal roads as rr
rr.compareAlgorithms(40)
```

# Dependencies #

mmh3, matplotlib, joblib