from roadrunner.randomness import consumer_seed, current_seed, random_seed, spawn_seeds


class TestConsumerSeed:
    def test_deterministic(self):
        assert consumer_seed(13, 2, "fit") == consumer_seed(13, 2, "fit")

    def test_differs_by_snapshot(self):
        assert consumer_seed(13, 2, "fit") != consumer_seed(13, 3, "fit")

    def test_differs_by_name(self):
        assert consumer_seed(13, 2, "fit") != consumer_seed(13, 2, "los")

    def test_near_miss_does_not_collide(self):
        # fit@5 and los@4 would collide under seed + snap + index; the
        # snapshot term must be multiplied by the consumer-count cap.
        assert consumer_seed(13, 5, "fit") != consumer_seed(13, 4, "los")


class TestSpawnSeeds:
    def test_deterministic(self):
        assert spawn_seeds(42, 4) == spawn_seeds(42, 4)

    def test_independent_looking(self):
        seeds = spawn_seeds(42, 5)
        assert len(set(seeds)) == 5

    def test_different_parent_different_children(self):
        assert spawn_seeds(42, 3) != spawn_seeds(43, 3)


class TestRandomSeedContext:
    def test_outside_context_returns_none(self):
        assert current_seed("fit") is None

    def test_inside_context_returns_value(self):
        with random_seed(fit=123, los=456):
            assert current_seed("fit") == 123
            assert current_seed("los") == 456

    def test_unset_name_returns_none(self):
        with random_seed(fit=123):
            assert current_seed("los") is None

    def test_restored_after_context(self):
        with random_seed(fit=123):
            pass
        assert current_seed("fit") is None

    def test_nestable(self):
        with random_seed(fit=1):
            with random_seed(fit=2):
                assert current_seed("fit") == 2
            assert current_seed("fit") == 1
