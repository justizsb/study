# Data Structures
## chapter 3: data structure overview
逻辑结构
<pre>
线性：
1. 数组 array
2. 链表 linked list
3. 栈 stack
4. 队列 queue

非线性：
树状 = 一对多
1. 树 tree
2. 堆 heap
3. 哈希表 hash table
网状 = 多对多
1. 图 graph
</pre>
物理结构
<pre>
连续 contiguous：array based, static data structure - length cannot be changed
1. stack
2. queue
3. hash table
4. tree
5. heap
6. graph
7. matrix
8. tensor (arrays with more than 3 dimensions)

分散 non-contiguous：linked list based, dynamic data structure - length can be changed
1. stack
2. queue
3. hash table
4. tree
5. heap
6. graph
</pre>
Basic data types: binary form - 1 byte = 8 bits
<pre>
integer type: byte, short, int, long
floating-point type: float, double
character type: char/str
boolean type: bool
</pre>
Number encoding
<pre>
原码 sign-magnitude = 最左的binary digit是1为负数 negative number，0为正数 positive number
反码 one‘s complement = 正数和原码相同；负数所有digit相反，除了最左
正码 two’s complement = 正数和原码相同；负数在反码上加1

计算：转成反码计算，再转回原码；基于加法计算，快速又简化硬件

float：
representation of float includes an exponent bit = much larger range than int, less precise
两头的bit有特殊意义，如0，∞，NaN
</pre>
Character encoding
<pre>
ASCII：english only, 7 digits = 128 char
EASCII: 8 digits = 256 char
GBK: 21886汉字
Unicode: 统一，多语言，same length, 2/3/4 bytes
UTF-8: variable-length encoding, 1-4 bytes
UTF-16: 2-4 bytes
UTF-32: 4 bytes

编程语言：UTF-16/32
1. random access
2. character counting O(1)
3. string operations 

UTF-16: surrogate pairs to represent unicode > 16 bits, losing advantage of fixed-length encoding, add complexity and difficulty
java, javascript, typescript, c#

unicode: python
UTF-8: go, rust
</pre>

## chapter 4: array and linked list
array
<pre>
memory address = array's memory address + item length * index
access: O(1)
insert: O(n), lose last item if fixed length
delete: O(n)
find: O(n), linear search
expand: O(n), if length is immutable

Pros:
Space efficiency: contiguous block of memory
Random access: O(1)
Cache locality: caches the surrounding data, speed up subsequent operations

Cons:
Low efficiency in insertion and deletion
Fixed length
Space wastage: extra space is wasted

Applications:
random access
sorting and searching
lookup table: relational retrieval
machine learning: linear algebra operations, neural network programming
data structure implementation: building blocks for stack, queue, hash table, heaps, graphs
</pre>
linked list
<pre>
non-contiguous memory

Common operations:
1. initialization: initialize node, create reference link, so can traverse to each node
2. insertion: O(1) - change n1 = n0.next, insertNode.next = n1, n0.next = insertNode
3. deletion: O(1) - deleteNode = n0.next, n1 = deleteNote.next, n0.next = n1
4. access: O(n) - need to traverse from first node
5. find: O(n) 

Common types:
1. singly linked list: pointer to next
2. circular linked list: tail node pointed to head node
3. doubly linked list: pointer to next and prev

Common applications:
Singly linked list - implementing data structures
1. stacks: insert and delete at the same end, last in first out
2. queues: insert and delete at the beginning, first in first out
3. hash table: chaining
4. graphs: adjaceny lists, each graph vertex with a linked list, each item in the linked list is vertices connected to the corresponding vertex
Doubly linked list - rapid access to preceding and succeeding elements
1. advanced data structures: red-black trees, B-trees, need to access node's parent
2. browser history: visted pages, click forward or back
3. least recently used algorithm: cache eviction, identify the least recently used
Circular linked list - periodic operations
1. round-robin scheduling algorithm:  common CPU scheduling method, cycling through a group of processes, fair and time-shared system among all processes
2. data buffers: divide data strean into multiple buffer blocks in circular fashion for seamless playback
</pre>
list
<pre>
based on array or linked list
variable length, dynamic array
array is a fixed length list

Common operations:
1. initialization: with or without initial values
2. access: O(1)
3. insert and delete: O(n)
4. concatenation: nums += nums1
5. sorting: after sorting, binary search or two-pointer
6. set: update item to a new value O(1)

list implementation:
1. initial capacity
2. size recording
3. expansaion mechanism 
</pre>
memory and cache
<pre>
hard disk: large data, long-term storage
random-access memory RAM: temporary storage, data being processed during execution
cache memory: frequently accessed data and instructions

degree of fragmentataion of free memory becomes higher when frequently allocated and released - linked list

cache hit rate: the fewer the cache misses, the higher CPU read-write efficiency
1. cache lines: transmission is more efficient than byte
2. prefetch mechanism: predict data access patterns, sequential/fixed stride jumping to improve hit rate
3. spatial locality: load nearby data
4. temporal locality: retain recently accessed data

cache utilization efficiency
1. occupied space: linked list more space, less effective data in cache
2. cache lines: load by line, more invalid data for linked list
3. prefetch mechanism: linked list is less predictable than array
4. spatial locality: array is stored in concentrated memory spaces

array has higher cache efficiency than linked list

</pre>
## chapter 5: stack and queue
stack
<pre>
first in last out 

Common operations
1. push to the top
2. pop from the top
3. peek the top element

Implementation
1. linked list: head insertion O(1)
2. array: push and pop from the end O(1)
time efficiency
1. array: efficiency decreases during expansion, average efficiency is higher
2. linked list: more stable efficiency
space efficiency
1. array: might waste some space
2. linked list: space occupied per node is relatively larger

Applications
1. back and foward in browsers: two stacks needed
2. undo and redo in software: two stacks needed
3. memory management in programs: record the function's context information, recursion keeps pushing onto stack, backtracking keeps popping from the stack
</pre>
queue
<pre>
first in first out

Common operations
1. push from the end
2. pop from the top
3. peek the top element

Implementation
1. linked list: head and tail, pop from head and push to tail
2. array: avoid O(n) deletion, use front, size, rear = front + size, to set access range; circular array = (i + capacity) % capacity

Applications
1. Amazon orders
2. Various to-do lists
</pre>
deque
<pre>
double-ended queue: can modify from both ends

Common operations
1. push_first: add to the top / appendleft in python
2. push_last: add to the end / append in python
3. pop_first: remove top element / popleft in python
4. pop_last: remove end element / pop in python
5. peek_first: access top element / [0] in python
6. peek_last: access end element / [-1] in python

Implementation
1. doubly linked list
2. array

Applications: all applications for queues and stacks
</pre>
## chapter 6: hash table
hash table
<pre>
{key: value} O(1)

Common operations
1. find element O(1)
2. add element O(1)
3. pop element O(1)

Implementation
index = hash(key) % capacity, where hash is some hash algorithm
</pre>
hash collision
<pre>
two keys point to one index -> expand capacity to reduce hash collision
load factor: number of items / number of buckets

separate chaining: change single element to linked list
1. search element: enter key -> get bucket -> traverse linked list to find key
2. add element: add to linked list after getting bucket
3. delete element: delete from linked list after getting the bucket
limitation: increased space usage, reduced query efficiency
change linked list to AVL tree or red-black tree to improve from O(n) to O(logn)

open addressing: multiple probes
1. linear probing: linear search for probing
insert element: if bucket has the element, find an empty bucket to insert
find element: if found hash collision, linear search until getting the element or None
cannot delete the element to make it stop searching the rest, use TOMBSTONE
replace found element with TOMSTONE location
clustering effect
2. Quadratic probing: may not probe the entire hash table
3. Double hashing: multiple has functions for probing
insert: if f1(x) has confict, try f2(x), untile an empty position is found
search: search in the same order of hash functions until the target element is found

Programming languages
Python: open addressing
Java: separate chaining
Go: separate chaining
</pre>
hash algorithm
<pre>
target
1. determinism: same output for the same input
2. high efficiency: fast computing
3. uniform distribution: key-value pairs evenly distributed in the table

Application
1. password storage: hash the password
2. data integrity check
unidirectionality 单向性, collision resistance 抗碰撞性, avalanche effect 雪崩效应

Implementation
1. addition
2. multiplication
3. XOR
4. rotation

Common hash algorithm
1. MD5, SHA-1 useless now
2. SHA-2, SHA-256 GOOD
3. SHA-3 ok
</pre>
## chapter 7: tree
binary tree
<pre>
all nodes have children, except for leaf nodes

Common terms
1. root node: no parent node
2. leaf node: no child node, both left and right are None
3. edge: the bridge between nodes
4. level of the node: root level = 1, +1 from root
5. degree of the node: the number of children for the node, [0,2] for binary tree
6. height of the tree: the number of edges from root to the farthest leaf
7. depth of the node: the number of edges from root to node
8. height of the node: the number of edges from node to the farthest leaf
    leaf's height = 0
    None's height = -1
Common operations
1. Initialization: initialize node then use pointer to structure
2. insert and delete: modify the pointer

Common binary tree
1. perfect binary tree:
leaf's degree = 0
others' degree = 2
total nodes = 2^(height+1)-1
2. complete binary tree:
only the last level is not completed, and None for some right leaves
3. fully binary tree:
degree = 0 or 2
4. balanced binary tree:
the difference in height for each subtree is not > 1
5. imbalanced binary tree:
like linked list, O(n)
</pre>
```python
# 1. breadth-first search or level-order traversal: from root to leaf, from left to right
queue = deque[TreeNode] = deque()
queue.append(root)
res = []
while queue:
    node = queue.popleft()
    res.append(node.val)
    if node.left is not None:
        queue.append(node.left)
    if node.right is not None:
        queue.append(node.right)
return res
```
```python
# 2. depth-first search or depth-first traveral: traverse to the leaf and back

def pre_order(root): # 前序遍历 root -> left subtree -> right subtree
    if root is None: return
    res.append(root.val)
    pre_order(root.left)
    pre_order(root.right)

def in_order(root): # left subtree -> root -> right subtree
    if root is None: return
    in_order(root.left)
    res.append(root.val)
    in_order(root.right)

def post_order(root): # left subtree -> right subtree -> root
    if root is None: return
    post_order(root.left)
    post_order(root.right)
    res.append(root.val)
```
array based binary tree
<pre>
Common types
1. perfect binary tree:
root = 0
node = i, left = 2i+1, right = 2i+2
2. any binary tree: add None to represent empty node
3. complete binary tree: don't have to store all None

Pros
1. contiguous memory, good cache, visit and traversal efficiency
2. save space, no need to store pointers
3. random access
Cons
1. cannot have a large tree
2. insert and delete low efficiency
3. if a lot of None, low space efficiency
</pre>
binary search tree
<pre>
Definition
1. values in left subtree < value of root < values in right subtree
2. any left/right subtree is also a binary search tree

Common operations
1. find a node O(log n)
if cur.val < num, go right
if cur.val > num, go left
if cur.val = num, return
2. insert a node O(log n)
search until leaf, insert
cannot have duplicate values
need to have pre as parent
3. delete node
need to keep sub nodes the same as binary search tree
degree = 0: None
degree = 1: replace value
degree = 2: find the replaceable value using in_order: find the right subtree's leftest leaf
4. in_order = ascending order

Common application
1. multi-level indexing: efficient search, insert, delete
2. data structure for sorting algorithm
3. store data streams to keep them ordered
</pre>
AVL tree
<pre>
avoid becoming linked list after frequently insert and remove

AVL = binary search tree = balanced binary tree = balanced binary search tree
balance factor: height(node.left) - height(node.right) [-1,1] for AVL
</pre>
```python
# right rotation: when left subtree is heavy
def right_rotate(node):
    child = node.left
    grand_child = child.right
    child.right = node
    node.left = grand_child

    self.update_height(node) 
    self.update_height(child)
    # max([self.height(node.left), self.height(node.right)]) + 1

    return child

# left rotation: when right subtree is heavy
def left_rotate(node):
    child = node.right
    grand_child = child.left
    child.left = node
    node.right = grand_child
    
    self.update_height(node)
    self.update_height(child)

    return child

def left_then_right(node):
    node.left = left_rotate(node.left)
    return right_rotate(node)

def right_then_left(node):
    node.right = right_rotate(node.right)
    return left_rotate(node)
```
| 失衡节点的平衡因子 | 子节点的平衡因子 | 应采用的旋转方法 |
| ------------------ | ---------------- | ---------------- |
| $> 1$ （左偏树）   | $\geq 0$         | 右旋             |
| $> 1$ （左偏树）   | $<0$             | 先左旋后右旋     |
| $< -1$ （右偏树）  | $\leq 0$         | 左旋             |
| $< -1$ （右偏树）  | $>0$             | 先右旋后左旋     |
```python
def insert(self, node, value):
    if node is None:
        return TreeNode(val)
    if val < node.val:
        node.left = self.insert(node.left, val)
    elif val > node.val:
        node.right = self.insert(node.right, val)

    self.update_height(node)

    return self.rotate(node)
```
<pre>
Common operations AVL:
1. insert: insert then rotate to adjust
2. delete: delete then rotate to adjust

Common applications AVL:
1. organize and save large data, high frequency search, low frequency delete
2. indexing in database
3. red-black tree is more popular, easier to balance with less rotation, more efficient to add/remove
</pre>
## chapter 8: heap
<pre>
complete binary tree with two special types
1. min heap: value of any node <= values of subnodes, minimum = root
2. max heap: value of any node >= values of subnodes, maximum = root
root = top, rightest leaf = bottom

Common operations:
heap is used to implement priority queue
1. push: enter the heap O(log n)
2. pop: pop the root O(log n)
3. peek: the value of the root
4. size: number of nodes
5. isEmpty: if the heap is empty
</pre>
```python
import heapq # default min heap

heap = []
heapq.heappush(heap, 1) # push item to heap
heapq.heappop(heap) # pop from heap
len(heap) # size
heap.heapify(heap) # enter a list and make it heap
```
<pre>
array based implementation
</pre>
```python
def left(i): return 2*i+1
def right(i): return 2*i+2
def parent(i): return (i-1)//2

def size(): 
    return len(heap)

def peek(): 
    return heap[0]

def swap(i, j):
    heap[i], heap[j] = heap[j], heap[i]

def push(val):
    heap.append(val)
    sift_up(size-1)

def sift_up(i): # from index i to root
    while True:
        p = parent(i) # parent's index
        if p < 0 or heap[i] <=/>= heap[p]: break
        swap(i, p)
        i = p

def pop(): # exchange root with rightest leaf, then sift down from root
    if is_empty: reutn IndexError()

    swap(0, size-1)
    val = heap.pop() # remove the last node with root's value
    sift_down(0)

def sift_down(i): # from top to down
    while True:
        left_index, right_index, max_index = left(i), right(i), i
        if left_index < size and heap[left_index] >/< heap[max_index]:
            max_index = left_index
        if right_index < size and heap[right_index] >/< heap[max_index]:
            max_index = right_index
        if max_index == i:
            break
        swap(i, max_index)
        i = max_index
```
<pre>
Common application
1. priority queue: push O(log n), pop O(log n), implement O(n)
2. heap sorting
3. Top k item
</pre>
```python
def __init__(nums): # O(n) to create heap
    self.heap = nums
    for i in range(parent(size-1), -1, -1): # from the last node sift down
        sift_down(i)
```
<pre>
Top-K
1. brute-force: each round found the top 1,2,...,k elem O(n^2)
2. sort: after sort then get top K elem O(n log n)
3. heap: init a min heap with k element, from k+1 element, compare and swap
</pre>
```python
def top_k_heap(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)
    for i in range(k, len(nums)):
        if nums[i] > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, nums[i])

    return heap

```
## chapter 9: graph

# Algorithms
## chapter 10: search
## chapter 11: sorting
## chapter 12: divide and conquer
## chapter 13: backtracking
## chapter 14: dynamic programming
## chapter 15: greedy