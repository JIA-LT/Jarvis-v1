# Python 编程指南

## 第一章：Python 基础

Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它由 Guido van Rossum 在 1991 年首次发布，现在已经成为最受欢迎的编程语言之一。

### 1.1 变量和数据类型

在 Python 中，变量不需要显式声明类型。你可以直接赋值：

```python
# 整数
age = 25
count = 100

# 浮点数
price = 99.99
temperature = 36.5

# 字符串
name = "Python"
message = 'Hello, World!'

# 布尔值
is_active = True
is_complete = False

# 列表
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]

# 字典
person = {
    "name": "Alice",
    "age": 30,
    "city": "Beijing"
}
```

### 1.2 控制流

Python 提供了多种控制流语句，包括条件语句和循环语句。

#### 条件语句

```python
# if-elif-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"你的成绩是: {grade}")
```

#### 循环语句

```python
# for 循环
for i in range(5):
    print(f"数字: {i}")

# while 循环
count = 0
while count < 5:
    print(f"计数: {count}")
    count += 1

# 列表推导式
squares = [x**2 for x in range(10)]
even_numbers = [x for x in range(20) if x % 2 == 0]
```

## 第二章：函数和模块

函数是 Python 中组织代码的基本单元。它们允许你将代码块封装成可重用的组件。

### 2.1 定义函数

```python
def greet(name):
    """简单的问候函数"""
    return f"Hello, {name}!"

def calculate_area(length, width):
    """计算矩形面积"""
    area = length * width
    return area

def process_data(data, operation="sum"):
    """处理数据，支持多种操作"""
    if operation == "sum":
        return sum(data)
    elif operation == "average":
        return sum(data) / len(data)
    elif operation == "max":
        return max(data)
    else:
        return None
```

### 2.2 高级函数特性

Python 支持多种高级函数特性：

- **默认参数**：函数参数可以有默认值
- **关键字参数**：可以按名称传递参数
- **可变参数**：使用 `*args` 和 `**kwargs`
- **Lambda 函数**：匿名函数
- **装饰器**：修改函数行为的强大工具

```python
# Lambda 函数
add = lambda x, y: x + y
multiply = lambda x, y: x * y

# 使用 map 和 filter
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 装饰器示例
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f} 秒")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "完成"
```

## 第三章：面向对象编程

Python 完全支持面向对象编程（OOP），包括类、继承、多态等特性。

### 3.1 类和对象

```python
class Person:
    """人员类"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"我是 {self.name}，今年 {self.age} 岁"
    
    def have_birthday(self):
        self.age += 1
        return f"生日快乐！现在 {self.age} 岁了"

# 创建对象
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

print(person1.introduce())
print(person2.introduce())
```

### 3.2 继承和多态

```python
class Animal:
    """动物基类"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("子类必须实现 speak 方法")

class Dog(Animal):
    """狗类"""
    
    def speak(self):
        return f"{self.name} 说: 汪汪！"

class Cat(Animal):
    """猫类"""
    
    def speak(self):
        return f"{self.name} 说: 喵喵！"

# 多态示例
animals = [Dog("旺财"), Cat("小花")]
for animal in animals:
    print(animal.speak())
```

## 第四章：数据处理

Python 在数据处理方面非常强大，特别是在科学计算和数据分析领域。

### 4.1 列表操作

```python
# 列表切片
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
first_three = numbers[:3]
last_three = numbers[-3:]
middle = numbers[3:7]

# 列表方法
fruits = ["apple", "banana"]
fruits.append("orange")  # 添加元素
fruits.insert(1, "grape")  # 在指定位置插入
fruits.remove("banana")  # 删除元素
fruits.sort()  # 排序
fruits.reverse()  # 反转
```

### 4.2 字典操作

```python
# 字典创建和访问
student = {
    "name": "张三",
    "age": 20,
    "grades": [85, 90, 88]
}

# 访问值
name = student["name"]
age = student.get("age", 0)  # 安全访问，有默认值

# 添加和修改
student["major"] = "计算机科学"
student["age"] = 21

# 遍历字典
for key, value in student.items():
    print(f"{key}: {value}")
```

## 第五章：文件操作

Python 提供了简单而强大的文件操作功能。

### 5.1 读取文件

```python
# 读取整个文件
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 逐行读取
with open("data.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())

# 读取所有行到列表
with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### 5.2 写入文件

```python
# 写入文件
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("第一行内容\n")
    f.write("第二行内容\n")

# 追加内容
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("追加的内容\n")
```

## 第六章：异常处理

Python 使用 try-except 块来处理异常，使程序更加健壮。

### 6.1 基本异常处理

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零！")

try:
    number = int("abc")
except ValueError:
    print("无法转换为整数")

try:
    file = open("nonexistent.txt", "r")
except FileNotFoundError:
    print("文件不存在")
```

### 6.2 高级异常处理

```python
try:
    # 可能出错的代码
    result = process_data(data)
except ValueError as e:
    print(f"值错误: {e}")
except TypeError as e:
    print(f"类型错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
else:
    print("一切正常")
finally:
    print("清理工作")
```

## 总结

Python 是一门功能强大且易于学习的编程语言。它适用于多种应用场景：

1. **Web 开发**：使用 Django、Flask 等框架
2. **数据科学**：使用 NumPy、Pandas、Matplotlib
3. **机器学习**：使用 TensorFlow、PyTorch、Scikit-learn
4. **自动化脚本**：系统管理和任务自动化
5. **游戏开发**：使用 Pygame 等库

掌握 Python 的基础知识后，你可以根据具体需求深入学习相关领域的库和框架。记住，编程是一门实践性很强的技能，多写代码、多练习是提高的关键。

祝你编程愉快！
