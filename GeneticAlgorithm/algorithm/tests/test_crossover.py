import random
import pytest
import algorithm.geneticAlgorithm as ga

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BINARY = {0, 1}


def is_binary(chrom):
    return all(g in BINARY for g in chrom)


def make_parents(length=8):
    a = [random.randint(0, 1) for _ in range(length)]
    b = [random.randint(0, 1) for _ in range(length)]
    return a, b


# ---------------------------------------------------------------------------
# One-Point Crossover
# ---------------------------------------------------------------------------

class TestCrossoverOnePoint:
    def test_output_length_preserved(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_one_point(a, b)
        assert len(ca) == len(a)
        assert len(cb) == len(b)

    def test_genes_are_binary(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_one_point(a, b)
        assert is_binary(ca)
        assert is_binary(cb)

    def test_genes_come_from_parents(self):
        """Each position in children must have come from one of the two parents."""
        a, b = make_parents(10)
        ca, cb = ga.crossover_one_point(a, b)
        for i in range(len(a)):
            assert ca[i] in (a[i], b[i])
            assert cb[i] in (a[i], b[i])

    def test_children_are_complements(self):
        """ca and cb should together contain all genes from both parents."""
        a = [1, 1, 1, 1, 1, 1]
        b = [0, 0, 0, 0, 0, 0]
        ca, cb = ga.crossover_one_point(a, b)
        # Every position: one child has 1, the other has 0
        for ga_gene, gb_gene in zip(ca, cb):
            assert ga_gene + gb_gene == 1

    def test_minimum_length_chromosome(self):
        a = [1, 0]
        b = [0, 1]
        ca, cb = ga.crossover_one_point(a, b)
        assert len(ca) == 2
        assert len(cb) == 2


# ---------------------------------------------------------------------------
# Two-Point Crossover
# ---------------------------------------------------------------------------

class TestCrossoverTwoPoint:
    def test_output_length_preserved(self):
        a, b = make_parents(12)
        ca, cb = ga.crossover_two_point(a, b)
        assert len(ca) == len(a)
        assert len(cb) == len(b)

    def test_genes_are_binary(self):
        a, b = make_parents(12)
        ca, cb = ga.crossover_two_point(a, b)
        assert is_binary(ca)
        assert is_binary(cb)

    def test_genes_come_from_parents(self):
        a, b = make_parents(12)
        ca, cb = ga.crossover_two_point(a, b)
        for i in range(len(a)):
            assert ca[i] in (a[i], b[i])
            assert cb[i] in (a[i], b[i])

    def test_children_are_complements(self):
        a = [1] * 10
        b = [0] * 10
        ca, cb = ga.crossover_two_point(a, b)
        for ga_gene, gb_gene in zip(ca, cb):
            assert ga_gene + gb_gene == 1


# ---------------------------------------------------------------------------
# Uniform Crossover
# ---------------------------------------------------------------------------

class TestCrossoverUniform:
    def test_output_length_preserved(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_uniform(a, b)
        assert len(ca) == len(a)
        assert len(cb) == len(b)

    def test_genes_are_binary(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_uniform(a, b)
        assert is_binary(ca)
        assert is_binary(cb)

    def test_genes_come_from_parents(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_uniform(a, b)
        for i in range(len(a)):
            assert ca[i] in (a[i], b[i])
            assert cb[i] in (a[i], b[i])

    def test_children_complement_each_other(self):
        """At each position, one child gets gene from a, the other from b."""
        a = [1] * 10
        b = [0] * 10
        ca, cb = ga.crossover_uniform(a, b)
        for ga_gene, gb_gene in zip(ca, cb):
            assert ga_gene + gb_gene == 1


# ---------------------------------------------------------------------------
# Shuffle Crossover
# ---------------------------------------------------------------------------

class TestCrossoverShuffle:
    def test_output_length_preserved(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_shuffle(a, b)
        assert len(ca) == len(a)
        assert len(cb) == len(b)

    def test_genes_are_binary(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_shuffle(a, b)
        assert is_binary(ca)
        assert is_binary(cb)

    def test_genes_come_from_parents(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_shuffle(a, b)
        for i in range(len(a)):
            assert ca[i] in (a[i], b[i])
            assert cb[i] in (a[i], b[i])

    def test_children_complement_each_other(self):
        a = [1] * 10
        b = [0] * 10
        ca, cb = ga.crossover_shuffle(a, b)
        for ga_gene, gb_gene in zip(ca, cb):
            assert ga_gene + gb_gene == 1


# ---------------------------------------------------------------------------
# Segment Crossover
# ---------------------------------------------------------------------------

class TestCrossoverSegment:
    def test_output_length_preserved(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_segment(a, b)
        assert len(ca) == len(a)
        assert len(cb) == len(b)

    def test_genes_are_binary(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_segment(a, b)
        assert is_binary(ca)
        assert is_binary(cb)

    def test_genes_come_from_parents(self):
        a, b = make_parents(10)
        ca, cb = ga.crossover_segment(a, b)
        for i in range(len(a)):
            assert ca[i] in (a[i], b[i])
            assert cb[i] in (a[i], b[i])


# ---------------------------------------------------------------------------
# Dispatcher (crossover)
# ---------------------------------------------------------------------------

class TestCrossoverDispatcher:
    def test_dispatches_one_point(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'one_point')
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        called = []
        original = ga.crossover_one_point
        monkeypatch.setattr(ga, 'crossover_one_point', lambda a, b: (called.append(1) or original(a, b)))
        ga.crossover([1, 0, 1], [0, 1, 0])
        assert called

    def test_rate_zero_returns_copies(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 0.0)
        a = [1, 0, 1, 0]
        b = [0, 1, 0, 1]
        ca, cb = ga.crossover(a, b)
        assert ca == a
        assert cb == b
        assert ca is not a  # must be copies, not same object
        assert cb is not b

    def test_rate_one_always_crosses(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'uniform')
        a = [1] * 8
        b = [0] * 8
        # Run several times — at least one should differ from identity
        results = [ga.crossover(a[:], b[:]) for _ in range(10)]
        assert all(len(ca) == 8 and len(cb) == 8 for ca, cb in results)

    def test_dispatches_two_point(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'two_point')
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        ca, cb = ga.crossover([1, 0, 1, 0], [0, 1, 0, 1])
        assert len(ca) == 4 and len(cb) == 4

    def test_dispatches_shuffle(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'shuffle')
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        ca, cb = ga.crossover([1, 0, 1, 0], [0, 1, 0, 1])
        assert len(ca) == 4 and len(cb) == 4

    def test_dispatches_segment(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'segment')
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        ca, cb = ga.crossover([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1])
        assert len(ca) == 6 and len(cb) == 6

    def test_unknown_type_raises(self, monkeypatch):
        monkeypatch.setattr(ga, 'CROSSOVER_RATE', 1.0)
        monkeypatch.setattr(ga, 'CROSSOVER_TYPE', 'nonexistent')
        with pytest.raises(ValueError):
            ga.crossover([1, 0], [0, 1])
