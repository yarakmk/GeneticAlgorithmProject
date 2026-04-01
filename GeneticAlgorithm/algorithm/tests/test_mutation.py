import pytest
import algorithm.geneticAlgorithm as ga

BINARY = {0, 1}


def is_binary(chrom):
    return all(g in BINARY for g in chrom)


# ---------------------------------------------------------------------------
# Bit-Flip Mutation
# ---------------------------------------------------------------------------

class TestMutateBitFlip:
    def test_output_length_preserved(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.1)
        chrom = [1, 0, 1, 0, 1, 0, 1, 0]
        result = ga.mutate_bit_flip(chrom)
        assert len(result) == len(chrom)

    def test_genes_are_binary(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.5)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_bit_flip(chrom)
        assert is_binary(result)

    def test_rate_zero_no_mutations(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0, 1, 1]
        result = ga.mutate_bit_flip(chrom)
        assert result == chrom

    def test_rate_one_all_flipped(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 1.0)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_bit_flip(chrom)
        expected = [1 - g for g in chrom]
        assert result == expected

    def test_input_not_mutated(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 1.0)
        chrom = [1, 0, 1, 0]
        original = chrom[:]
        ga.mutate_bit_flip(chrom)
        assert chrom == original  # original must be untouched


# ---------------------------------------------------------------------------
# Gauss-by-Center Mutation
# ---------------------------------------------------------------------------

class TestMutateGaussByCenter:
    def test_output_length_preserved(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.1)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_gauss_by_center(chrom)
        assert len(result) == len(chrom)

    def test_genes_are_binary(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.5)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_gauss_by_center(chrom)
        assert is_binary(result)

    def test_rate_zero_preserves_genes(self, monkeypatch):
        """With std=0 noise is always 0 so genes should never change."""
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_gauss_by_center(chrom)
        assert result == chrom

    def test_output_always_valid_over_many_runs(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 2.0)  # very high noise
        chrom = [1, 0, 1, 0, 1, 0]
        for _ in range(50):
            result = ga.mutate_gauss_by_center(chrom)
            assert is_binary(result)


# ---------------------------------------------------------------------------
# Uniform Mutation
# ---------------------------------------------------------------------------

class TestMutateUniform:
    def test_output_length_preserved(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.3)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_uniform(chrom)
        assert len(result) == len(chrom)

    def test_genes_are_binary(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.5)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_uniform(chrom)
        assert is_binary(result)

    def test_rate_zero_no_mutations(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0, 1, 0]
        result = ga.mutate_uniform(chrom)
        assert result == chrom


# ---------------------------------------------------------------------------
# Dispatcher (mutate)
# ---------------------------------------------------------------------------

class TestMutateDispatcher:
    def test_dispatches_bit_flip(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_TYPE', 'bit_flip')
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0]
        assert ga.mutate(chrom) == chrom

    def test_dispatches_gauss_by_center(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_TYPE', 'gauss_by_center')
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0]
        result = ga.mutate(chrom)
        assert len(result) == len(chrom)

    def test_dispatches_uniform_mutation(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_TYPE', 'uniform_mutation')
        monkeypatch.setattr(ga, 'MUTATION_RATE', 0.0)
        chrom = [1, 0, 1, 0]
        assert ga.mutate(chrom) == chrom

    def test_unknown_type_raises(self, monkeypatch):
        monkeypatch.setattr(ga, 'MUTATION_TYPE', 'nonexistent')
        with pytest.raises(ValueError):
            ga.mutate([1, 0, 1])
