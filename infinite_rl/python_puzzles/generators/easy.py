"""Easy programming puzzles - intermediate difficulty between math (level 0) and programming puzzles (level 1)
These puzzles introduce basic programming concepts with simpler logic than standard programming puzzles."""

from ..puzzle_generator import PuzzleGenerator, Tags
from typing import List


# 1. String manipulation - basic
class ReverseString(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="hello"):
        """Reverse a given string."""
        return x == s[::-1]

    @staticmethod
    def sol(s):
        return s[::-1]

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 20))])
        self.add(dict(s=s))


class RepeatString(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="abc", n=5):
        """Repeat a string n times."""
        return x == s * n

    @staticmethod
    def sol(s, n):
        return s * n

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 5))])
        n = self.random.randint(1, 20)
        self.add(dict(s=s, n=n))


class StringLength(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, s="programming"):
        """Find the length of a string."""
        return x == len(s)

    @staticmethod
    def sol(s):
        return len(s)

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 50))])
        self.add(dict(s=s))


class Uppercase(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="hello world"):
        """Convert a string to uppercase."""
        return x == s.upper()

    @staticmethod
    def sol(s):
        return s.upper()

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                     for _ in range(self.random.randint(5, 30))])
        self.add(dict(s=s))


class Lowercase(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="HELLO WORLD"):
        """Convert a string to lowercase."""
        return x == s.lower()

    @staticmethod
    def sol(s):
        return s.lower()

    def gen_random(self):
        s = "".join([chr(self.random.randint(65, 90)) if self.random.random() > 0.2 else " " 
                     for _ in range(self.random.randint(5, 30))])
        self.add(dict(s=s))


# 2. List operations - basic
class ListSum(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[1, 2, 3, 4, 5]):
        """Find the sum of a list of integers."""
        return x == sum(nums)

    @staticmethod
    def sol(nums):
        return sum(nums)

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


class ListMax(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[3, 7, 2, 9, 1]):
        """Find the maximum value in a list."""
        return x == max(nums)

    @staticmethod
    def sol(nums):
        return max(nums)

    def gen_random(self):
        nums = [self.random.randint(-1000, 1000) for _ in range(self.random.randint(1, 50))]
        self.add(dict(nums=nums))


class ListMin(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[3, 7, 2, 9, 1]):
        """Find the minimum value in a list."""
        return x == min(nums)

    @staticmethod
    def sol(nums):
        return min(nums)

    def gen_random(self):
        nums = [self.random.randint(-1000, 1000) for _ in range(self.random.randint(1, 50))]
        self.add(dict(nums=nums))


class ListLength(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[1, 2, 3, 4, 5]):
        """Find the length of a list."""
        return x == len(nums)

    @staticmethod
    def sol(nums):
        return len(nums)

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(0, 100))]
        self.add(dict(nums=nums))


class ReverseList(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5]):
        """Reverse a list."""
        return x == nums[::-1]

    @staticmethod
    def sol(nums):
        return nums[::-1]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


# 3. Simple counting and filtering
class CountPositive(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[1, -2, 3, -4, 5]):
        """Count the number of positive integers in a list."""
        return x == len([n for n in nums if n > 0])

    @staticmethod
    def sol(nums):
        return len([n for n in nums if n > 0])

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 50))]
        self.add(dict(nums=nums))


class CountEven(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[1, 2, 3, 4, 5, 6]):
        """Count the number of even integers in a list."""
        return x == len([n for n in nums if n % 2 == 0])

    @staticmethod
    def sol(nums):
        return len([n for n in nums if n % 2 == 0])

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 50))]
        self.add(dict(nums=nums))


class CountOdd(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, nums=[1, 2, 3, 4, 5, 6]):
        """Count the number of odd integers in a list."""
        return x == len([n for n in nums if n % 2 != 0])

    @staticmethod
    def sol(nums):
        return len([n for n in nums if n % 2 != 0])

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 50))]
        self.add(dict(nums=nums))


class FilterPositive(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, -2, 3, -4, 5]):
        """Return a list containing only positive numbers."""
        return x == [n for n in nums if n > 0]

    @staticmethod
    def sol(nums):
        return [n for n in nums if n > 0]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


class FilterEven(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5, 6]):
        """Return a list containing only even numbers."""
        return x == [n for n in nums if n % 2 == 0]

    @staticmethod
    def sol(nums):
        return [n for n in nums if n % 2 == 0]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


# 4. Simple arithmetic operations
class MultiplyList(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3], k=5):
        """Multiply each element in the list by k."""
        return x == [n * k for n in nums]

    @staticmethod
    def sol(nums, k):
        return [n * k for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(1, 20))]
        k = self.random.randint(-10, 10)
        self.add(dict(nums=nums, k=k))


class AddConstant(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3], k=10):
        """Add k to each element in the list."""
        return x == [n + k for n in nums]

    @staticmethod
    def sol(nums, k):
        return [n + k for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(1, 20))]
        k = self.random.randint(-100, 100)
        self.add(dict(nums=nums, k=k))


class SquareList(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4]):
        """Square each element in the list."""
        return x == [n * n for n in nums]

    @staticmethod
    def sol(nums):
        return [n * n for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-20, 20) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


class AbsoluteValues(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[-1, 2, -3, 4]):
        """Return the absolute value of each element."""
        return x == [abs(n) for n in nums]

    @staticmethod
    def sol(nums):
        return [abs(n) for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


# 5. Range operations
class CreateRange(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], n=10):
        """Create a list of integers from 0 to n-1."""
        return x == list(range(n))

    @staticmethod
    def sol(n):
        return list(range(n))

    def gen_random(self):
        n = self.random.randint(1, 100)
        self.add(dict(n=n))


class RangeWithStart(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], start=5, end=15):
        """Create a list of integers from start to end-1."""
        return x == list(range(start, end))

    @staticmethod
    def sol(start, end):
        return list(range(start, end))

    def gen_random(self):
        start = self.random.randint(-50, 50)
        end = self.random.randint(start + 1, start + 100)
        self.add(dict(start=start, end=end))


class RangeWithStep(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], start=0, end=20, step=3):
        """Create a list of integers from start to end-1 with step."""
        return x == list(range(start, end, step))

    @staticmethod
    def sol(start, end, step):
        return list(range(start, end, step))

    def gen_random(self):
        start = self.random.randint(-50, 50)
        step = self.random.randint(1, 10)
        end = self.random.randint(start + step, start + 100)
        self.add(dict(start=start, end=end, step=step))


# 6. String operations
class JoinStrings(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, words=["hello", "world"], sep=" "):
        """Join a list of strings with a separator."""
        return x == sep.join(words)

    @staticmethod
    def sol(words, sep):
        return sep.join(words)

    def gen_random(self):
        words = ["".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))]) 
                 for _ in range(self.random.randint(1, 10))]
        sep = self.random.choice([" ", ",", "-", ""])
        self.add(dict(words=words, sep=sep))


class SplitString(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[str], s="hello world how are you", sep=" "):
        """Split a string by a separator."""
        return x == s.split(sep)

    @staticmethod
    def sol(s, sep):
        return s.split(sep)

    def gen_random(self):
        sep = self.random.choice([" ", ",", "-"])
        words = ["".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))]) 
                 for _ in range(self.random.randint(1, 10))]
        s = sep.join(words)
        self.add(dict(s=s, sep=sep))


class CountSubstring(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, s="hello hello world", sub="hello"):
        """Count occurrences of a substring."""
        return x == s.count(sub)

    @staticmethod
    def sol(s, sub):
        return s.count(sub)

    def gen_random(self):
        sub = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 5))])
        base = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(10, 30))])
        # Insert substring a few times
        parts = [base[i:i+5] for i in range(0, len(base), 5)]
        for i in range(self.random.randint(0, 3)):
            parts.insert(self.random.randint(0, len(parts)), sub)
        s = "".join(parts)
        self.add(dict(s=s, sub=sub))


class ReplaceString(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="hello world", old="world", new="python"):
        """Replace all occurrences of old with new."""
        return x == s.replace(old, new)

    @staticmethod
    def sol(s, old, new):
        return s.replace(old, new)

    def gen_random(self):
        old = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(2, 5))])
        new = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(2, 5))])
        base = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.3 else " " 
                       for _ in range(self.random.randint(10, 30))])
        parts = base.split(" ")
        for i in range(len(parts)):
            if self.random.random() < 0.3:
                parts[i] = old
        s = " ".join(parts)
        self.add(dict(s=s, old=old, new=new))


class StartsWith(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, s="hello world", prefix="hello"):
        """Check if string starts with prefix."""
        return x == s.startswith(prefix)

    @staticmethod
    def sol(s, prefix):
        return s.startswith(prefix)

    def gen_random(self):
        if self.random.random() < 0.5:
            # True case
            prefix = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))])
            rest = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                           for _ in range(self.random.randint(5, 20))])
            s = prefix + rest
        else:
            # False case
            s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                        for _ in range(self.random.randint(10, 30))])
            prefix = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))])
        self.add(dict(s=s, prefix=prefix))


class EndsWith(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, s="hello world", suffix="world"):
        """Check if string ends with suffix."""
        return x == s.endswith(suffix)

    @staticmethod
    def sol(s, suffix):
        return s.endswith(suffix)

    def gen_random(self):
        if self.random.random() < 0.5:
            # True case
            suffix = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))])
            rest = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                           for _ in range(self.random.randint(5, 20))])
            s = rest + suffix
        else:
            # False case
            s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                        for _ in range(self.random.randint(10, 30))])
            suffix = "".join([chr(self.random.randint(97, 122)) for _ in range(self.random.randint(1, 10))])
        self.add(dict(s=s, suffix=suffix))


# 7. Simple logic
class IsEven(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: bool, n=42):
        """Check if a number is even."""
        return x == (n % 2 == 0)

    @staticmethod
    def sol(n):
        return n % 2 == 0

    def gen_random(self):
        n = self.random.randint(-1000, 1000)
        self.add(dict(n=n))


class IsPositive(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: bool, n=5):
        """Check if a number is positive."""
        return x == (n > 0)

    @staticmethod
    def sol(n):
        return n > 0

    def gen_random(self):
        n = self.random.randint(-1000, 1000)
        self.add(dict(n=n))


class InRange(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: bool, n=15, low=10, high=20):
        """Check if n is in range [low, high)."""
        return x == (low <= n < high)

    @staticmethod
    def sol(n, low, high):
        return low <= n < high

    def gen_random(self):
        low = self.random.randint(-100, 100)
        high = self.random.randint(low + 1, low + 100)
        n = self.random.randint(low - 10, high + 10)
        self.add(dict(n=n, low=low, high=high))


# 8. List indexing and slicing
class FirstElement(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[5, 2, 8, 1, 9]):
        """Get the first element of a list."""
        return x == nums[0]

    @staticmethod
    def sol(nums):
        return nums[0]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


class LastElement(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[5, 2, 8, 1, 9]):
        """Get the last element of a list."""
        return x == nums[-1]

    @staticmethod
    def sol(nums):
        return nums[-1]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


class GetSlice(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5], start=1, end=4):
        """Get a slice of the list from start to end."""
        return x == nums[start:end]

    @staticmethod
    def sol(nums, start, end):
        return nums[start:end]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(5, 30))]
        start = self.random.randint(0, len(nums) - 2)
        end = self.random.randint(start + 1, len(nums))
        self.add(dict(nums=nums, start=start, end=end))


class EveryNth(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5, 6, 7, 8], n=2):
        """Get every nth element from the list."""
        return x == nums[::n]

    @staticmethod
    def sol(nums, n):
        return nums[::n]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(5, 30))]
        n = self.random.randint(2, 5)
        self.add(dict(nums=nums, n=n))


# 9. List comprehension variations
class DoubleEvens(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5, 6]):
        """Double only the even numbers in the list."""
        return x == [n * 2 if n % 2 == 0 else n for n in nums]

    @staticmethod
    def sol(nums):
        return [n * 2 if n % 2 == 0 else n for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(5, 25))]
        self.add(dict(nums=nums))


class ZeroNegatives(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, -2, 3, -4, 5]):
        """Replace negative numbers with 0."""
        return x == [n if n >= 0 else 0 for n in nums]

    @staticmethod
    def sol(nums):
        return [n if n >= 0 else 0 for n in nums]

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(5, 25))]
        self.add(dict(nums=nums))


class ClampValues(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 15, 8, 25, 3], min_val=5, max_val=20):
        """Clamp values to be within [min_val, max_val]."""
        return x == [max(min_val, min(max_val, n)) for n in nums]

    @staticmethod
    def sol(nums, min_val, max_val):
        return [max(min_val, min(max_val, n)) for n in nums]

    def gen_random(self):
        min_val = self.random.randint(-50, 50)
        max_val = self.random.randint(min_val + 10, min_val + 100)
        nums = [self.random.randint(min_val - 50, max_val + 50) for _ in range(self.random.randint(5, 25))]
        self.add(dict(nums=nums, min_val=min_val, max_val=max_val))


# 10. Simple set operations
class UniqueElements(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 2, 3, 3, 3, 4]):
        """Return list with unique elements (order preserved)."""
        seen = set()
        expected = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                expected.append(n)
        return x == expected

    @staticmethod
    def sol(nums):
        seen = set()
        result = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

    def gen_random(self):
        base = [self.random.randint(-20, 20) for _ in range(self.random.randint(3, 10))]
        nums = []
        for _ in range(self.random.randint(10, 30)):
            nums.append(self.random.choice(base))
        self.add(dict(nums=nums))


class CountUnique(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[1, 2, 2, 3, 3, 3, 4]):
        """Count the number of unique elements."""
        return x == len(set(nums))

    @staticmethod
    def sol(nums):
        return len(set(nums))

    def gen_random(self):
        base = [self.random.randint(-20, 20) for _ in range(self.random.randint(3, 10))]
        nums = []
        for _ in range(self.random.randint(10, 30)):
            nums.append(self.random.choice(base))
        self.add(dict(nums=nums))


class HasDuplicates(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[1, 2, 3, 4, 5]):
        """Check if the list has any duplicates."""
        return x == (len(nums) != len(set(nums)))

    @staticmethod
    def sol(nums):
        return len(nums) != len(set(nums))

    def gen_random(self):
        if self.random.random() < 0.5:
            # With duplicates
            base = [self.random.randint(-20, 20) for _ in range(self.random.randint(3, 10))]
            nums = []
            for _ in range(self.random.randint(10, 30)):
                nums.append(self.random.choice(base))
        else:
            # Without duplicates
            nums = list(range(self.random.randint(5, 30)))
            self.random.shuffle(nums)
        self.add(dict(nums=nums))


# 11. Simple tuple/list operations
class PairElements(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[List[int]], nums=[1, 2, 3, 4]):
        """Pair consecutive elements."""
        return x == [[nums[i], nums[i+1]] for i in range(len(nums)-1)]

    @staticmethod
    def sol(nums):
        return [[nums[i], nums[i+1]] for i in range(len(nums)-1)]

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(2, 20))]
        self.add(dict(nums=nums))


class ZipLists(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[List[int]], a=[1, 2, 3], b=[4, 5, 6]):
        """Zip two lists together."""
        return x == [[a[i], b[i]] for i in range(len(a))]

    @staticmethod
    def sol(a, b):
        return [[a[i], b[i]] for i in range(len(a))]

    def gen_random(self):
        length = self.random.randint(1, 20)
        a = [self.random.randint(-50, 50) for _ in range(length)]
        b = [self.random.randint(-50, 50) for _ in range(length)]
        self.add(dict(a=a, b=b))


class SumPairs(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], a=[1, 2, 3], b=[4, 5, 6]):
        """Add corresponding elements from two lists."""
        return x == [a[i] + b[i] for i in range(len(a))]

    @staticmethod
    def sol(a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def gen_random(self):
        length = self.random.randint(1, 20)
        a = [self.random.randint(-50, 50) for _ in range(length)]
        b = [self.random.randint(-50, 50) for _ in range(length)]
        self.add(dict(a=a, b=b))


# 12. Simple boolean operations
class AllPositive(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[1, 2, 3, 4, 5]):
        """Check if all numbers are positive."""
        return x == all(n > 0 for n in nums)

    @staticmethod
    def sol(nums):
        return all(n > 0 for n in nums)

    def gen_random(self):
        if self.random.random() < 0.5:
            # All positive
            nums = [self.random.randint(1, 100) for _ in range(self.random.randint(1, 20))]
        else:
            # At least one non-positive
            nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


class AnyNegative(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[1, 2, -3, 4, 5]):
        """Check if any number is negative."""
        return x == any(n < 0 for n in nums)

    @staticmethod
    def sol(nums):
        return any(n < 0 for n in nums)

    def gen_random(self):
        if self.random.random() < 0.5:
            # Has negative
            nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 20))]
            nums[self.random.randint(0, len(nums)-1)] = self.random.randint(-100, -1)
        else:
            # All non-negative
            nums = [self.random.randint(0, 100) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


class AllEven(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[2, 4, 6, 8]):
        """Check if all numbers are even."""
        return x == all(n % 2 == 0 for n in nums)

    @staticmethod
    def sol(nums):
        return all(n % 2 == 0 for n in nums)

    def gen_random(self):
        if self.random.random() < 0.5:
            # All even
            nums = [self.random.randint(-50, 50) * 2 for _ in range(self.random.randint(1, 20))]
        else:
            # At least one odd
            nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


# 13. Character/ASCII operations
class CharToAscii(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, c="A"):
        """Get the ASCII value of a character."""
        return x == ord(c)

    @staticmethod
    def sol(c):
        return ord(c)

    def gen_random(self):
        c = chr(self.random.randint(32, 126))
        self.add(dict(c=c))


class AsciiToChar(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, n=65):
        """Convert ASCII value to character."""
        return x == chr(n)

    @staticmethod
    def sol(n):
        return chr(n)

    def gen_random(self):
        n = self.random.randint(32, 126)
        self.add(dict(n=n))


class IsLetter(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, c="a"):
        """Check if character is a letter."""
        return x == c.isalpha()

    @staticmethod
    def sol(c):
        return c.isalpha()

    def gen_random(self):
        if self.random.random() < 0.5:
            c = chr(self.random.randint(97, 122))  # lowercase
        else:
            c = chr(self.random.randint(48, 57))  # digit
        self.add(dict(c=c))


class IsDigit(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, c="5"):
        """Check if character is a digit."""
        return x == c.isdigit()

    @staticmethod
    def sol(c):
        return c.isdigit()

    def gen_random(self):
        if self.random.random() < 0.5:
            c = chr(self.random.randint(48, 57))  # digit
        else:
            c = chr(self.random.randint(97, 122))  # lowercase
        self.add(dict(c=c))


# 14. Simple sorting operations
class SortList(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[3, 1, 4, 1, 5, 9, 2, 6]):
        """Sort a list in ascending order."""
        return x == sorted(nums)

    @staticmethod
    def sol(nums):
        return sorted(nums)

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


class SortDescending(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[3, 1, 4, 1, 5, 9, 2, 6]):
        """Sort a list in descending order."""
        return x == sorted(nums, reverse=True)

    @staticmethod
    def sol(nums):
        return sorted(nums, reverse=True)

    def gen_random(self):
        nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(1, 30))]
        self.add(dict(nums=nums))


class IsSorted(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[1, 2, 3, 4, 5]):
        """Check if list is sorted in ascending order."""
        return x == (nums == sorted(nums))

    @staticmethod
    def sol(nums):
        return nums == sorted(nums)

    def gen_random(self):
        if self.random.random() < 0.5:
            # Sorted
            nums = sorted([self.random.randint(-100, 100) for _ in range(self.random.randint(1, 20))])
        else:
            # Not sorted
            nums = [self.random.randint(-100, 100) for _ in range(self.random.randint(2, 20))]
        self.add(dict(nums=nums))


# 15. Simple find operations
class FindIndex(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[1, 2, 3, 4, 5], target=3):
        """Find the index of target in the list."""
        return x == nums.index(target)

    @staticmethod
    def sol(nums, target):
        return nums.index(target)

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(5, 25))]
        target = self.random.choice(nums)
        self.add(dict(nums=nums, target=target))


class Contains(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: bool, nums=[1, 2, 3, 4, 5], target=3):
        """Check if target is in the list."""
        return x == (target in nums)

    @staticmethod
    def sol(nums, target):
        return target in nums

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(5, 25))]
        if self.random.random() < 0.5:
            target = self.random.choice(nums)
        else:
            target = self.random.randint(-100, 100)
        self.add(dict(nums=nums, target=target))


class CountOccurrences(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, nums=[1, 2, 2, 3, 2, 4], target=2):
        """Count how many times target appears in the list."""
        return x == nums.count(target)

    @staticmethod
    def sol(nums, target):
        return nums.count(target)

    def gen_random(self):
        base = [self.random.randint(-20, 20) for _ in range(self.random.randint(3, 10))]
        nums = []
        for _ in range(self.random.randint(10, 30)):
            nums.append(self.random.choice(base))
        target = self.random.choice(base)
        self.add(dict(nums=nums, target=target))


# 16. Simple number operations
class Factorial(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, n=5):
        """Calculate factorial of n."""
        result = 1
        for i in range(1, n + 1):
            result *= i
        return x == result

    @staticmethod
    def sol(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def gen_random(self):
        n = self.random.randint(0, 15)
        self.add(dict(n=n))


class Power(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, base=2, exp=10):
        """Calculate base raised to the power of exp."""
        return x == base ** exp

    @staticmethod
    def sol(base, exp):
        return base ** exp

    def gen_random(self):
        base = self.random.randint(-10, 10)
        exp = self.random.randint(0, 10)
        self.add(dict(base=base, exp=exp))


class Modulo(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, a=17, b=5):
        """Calculate a modulo b."""
        return x == a % b

    @staticmethod
    def sol(a, b):
        return a % b

    def gen_random(self):
        b = self.random.randint(1, 100)
        a = self.random.randint(-1000, 1000)
        self.add(dict(a=a, b=b))


class IntegerDivision(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, a=17, b=5):
        """Calculate integer division a // b."""
        return x == a // b

    @staticmethod
    def sol(a, b):
        return a // b

    def gen_random(self):
        b = self.random.randint(1, 100)
        a = self.random.randint(-1000, 1000)
        self.add(dict(a=a, b=b))


# 17. String character operations
class CountVowels(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, s="hello world"):
        """Count the number of vowels in a string."""
        vowels = "aeiouAEIOU"
        return x == sum(1 for c in s if c in vowels)

    @staticmethod
    def sol(s):
        vowels = "aeiouAEIOU"
        return sum(1 for c in s if c in vowels)

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                     for _ in range(self.random.randint(10, 40))])
        self.add(dict(s=s))


class CountConsonants(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: int, s="hello world"):
        """Count the number of consonants in a string."""
        vowels = "aeiouAEIOU"
        return x == sum(1 for c in s if c.isalpha() and c not in vowels)

    @staticmethod
    def sol(s):
        vowels = "aeiouAEIOU"
        return sum(1 for c in s if c.isalpha() and c not in vowels)

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                     for _ in range(self.random.randint(10, 40))])
        self.add(dict(s=s))


class RemoveSpaces(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="hello world  test"):
        """Remove all spaces from a string."""
        return x == s.replace(" ", "")

    @staticmethod
    def sol(s):
        return s.replace(" ", "")

    def gen_random(self):
        s = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.3 else " " 
                     for _ in range(self.random.randint(10, 40))])
        self.add(dict(s=s))


class StripWhitespace(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: str, s="  hello world  "):
        """Remove leading and trailing whitespace."""
        return x == s.strip()

    @staticmethod
    def sol(s):
        return s.strip()

    def gen_random(self):
        spaces_before = " " * self.random.randint(0, 10)
        spaces_after = " " * self.random.randint(0, 10)
        middle = "".join([chr(self.random.randint(97, 122)) if self.random.random() > 0.2 else " " 
                         for _ in range(self.random.randint(5, 20))])
        s = spaces_before + middle + spaces_after
        self.add(dict(s=s))


# 18. List construction
class RepeatElements(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3], times=3):
        """Repeat each element in the list times."""
        return x == [n for n in nums for _ in range(times)]

    @staticmethod
    def sol(nums, times):
        return [n for n in nums for _ in range(times)]

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(1, 10))]
        times = self.random.randint(1, 5)
        self.add(dict(nums=nums, times=times))


class FlattenOnce(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], nested=[[1, 2], [3, 4], [5]]):
        """Flatten a nested list by one level."""
        return x == [item for sublist in nested for item in sublist]

    @staticmethod
    def sol(nested):
        return [item for sublist in nested for item in sublist]

    def gen_random(self):
        nested = []
        for _ in range(self.random.randint(1, 10)):
            sublist = [self.random.randint(-50, 50) for _ in range(self.random.randint(0, 5))]
            nested.append(sublist)
        self.add(dict(nested=nested))


class AlternateElements(PuzzleGenerator):
    tags = [Tags.trivial]

    @staticmethod
    def sat(x: List[int], a=[1, 2, 3], b=[4, 5, 6]):
        """Alternate elements from two lists."""
        result = []
        for i in range(len(a)):
            result.append(a[i])
            result.append(b[i])
        return x == result

    @staticmethod
    def sol(a, b):
        result = []
        for i in range(len(a)):
            result.append(a[i])
            result.append(b[i])
        return result

    def gen_random(self):
        length = self.random.randint(1, 15)
        a = [self.random.randint(-50, 50) for _ in range(length)]
        b = [self.random.randint(-50, 50) for _ in range(length)]
        self.add(dict(a=a, b=b))


# 19. Simple accumulation
class CumulativeSum(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[1, 2, 3, 4, 5]):
        """Create cumulative sum list."""
        result = []
        total = 0
        for n in nums:
            total += n
            result.append(total)
        return x == result

    @staticmethod
    def sol(nums):
        result = []
        total = 0
        for n in nums:
            total += n
            result.append(total)
        return result

    def gen_random(self):
        nums = [self.random.randint(-20, 20) for _ in range(self.random.randint(1, 20))]
        self.add(dict(nums=nums))


class RunningMax(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[3, 1, 4, 1, 5, 9, 2]):
        """Create list of running maximum values."""
        result = []
        current_max = nums[0]
        for n in nums:
            current_max = max(current_max, n)
            result.append(current_max)
        return x == result

    @staticmethod
    def sol(nums):
        result = []
        current_max = nums[0]
        for n in nums:
            current_max = max(current_max, n)
            result.append(current_max)
        return result

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(1, 25))]
        self.add(dict(nums=nums))


class RunningMin(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: List[int], nums=[3, 1, 4, 1, 5, 9, 2]):
        """Create list of running minimum values."""
        result = []
        current_min = nums[0]
        for n in nums:
            current_min = min(current_min, n)
            result.append(current_min)
        return x == result

    @staticmethod
    def sol(nums):
        result = []
        current_min = nums[0]
        for n in nums:
            current_min = min(current_min, n)
            result.append(current_min)
        return result

    def gen_random(self):
        nums = [self.random.randint(-50, 50) for _ in range(self.random.randint(1, 25))]
        self.add(dict(nums=nums))


# 20. Simple comparisons
class MaxOfTwo(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, a=5, b=8):
        """Return the maximum of two numbers."""
        return x == max(a, b)

    @staticmethod
    def sol(a, b):
        return max(a, b)

    def gen_random(self):
        a = self.random.randint(-100, 100)
        b = self.random.randint(-100, 100)
        self.add(dict(a=a, b=b))


class MinOfTwo(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, a=5, b=8):
        """Return the minimum of two numbers."""
        return x == min(a, b)

    @staticmethod
    def sol(a, b):
        return min(a, b)

    def gen_random(self):
        a = self.random.randint(-100, 100)
        b = self.random.randint(-100, 100)
        self.add(dict(a=a, b=b))


class ThreeWayMax(PuzzleGenerator):
    tags = [Tags.math]

    @staticmethod
    def sat(x: int, a=5, b=8, c=3):
        """Return the maximum of three numbers."""
        return x == max(a, b, c)

    @staticmethod
    def sol(a, b, c):
        return max(a, b, c)

    def gen_random(self):
        a = self.random.randint(-100, 100)
        b = self.random.randint(-100, 100)
        c = self.random.randint(-100, 100)
        self.add(dict(a=a, b=b, c=c))
