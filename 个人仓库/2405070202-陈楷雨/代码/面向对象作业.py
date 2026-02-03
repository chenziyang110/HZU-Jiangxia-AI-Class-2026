class Staff:
    """员工基类（老师、普通员工的父类）"""
    def __init__(self,name,age,position,salary,dept):
        self.name = name
        self.age = age
        self.position = position
        self.salary = salary
        self.dept = dept

class Teacher(Staff):
    """讲师类（继承员工类）"""
    def __init__(self,name,age,salary,dept,school):
        super().__init__(name,age,'讲师',salary,dept)
        self.school = school
        self.teaching_course = []
    def teaching(self,class_object):
        self.teaching_course.append(class_object)
        class_object.teacher = self.name #绑定老师
        self.school.staffs.append(self) #讲师加入分校员工列表
        print(f"讲师{self.name}开始负责{class_object.class_num}班级")

class Course:
    '''课程类'''
    def __init__(self,name,price,outline):
        self.name = name
        self.price = price
        self.outline = outline

class Class:
    '''班级类'''
    def __init__(self,class_num,course,school):
        self.class_num = class_num #班级编号
        self.course = course  #课程对象
        self.school = school  #所属校区
        self.teacher = None  # 负责该班级的讲师
        self.students = []  # 班级中的学生列表
    
    def create_teaching_record(self):
        '''上课记录'''
        
        if self.teacher:
            print(f"为{self.class_num}班级创建教学记录，负责老师：{self.teacher}")
        else:
            print(f"{self.class_num}班级尚未分配老师，无法创建教学记录")
    
    def drop_out(self,student):
        '''学员退学'''
        if student in self.students:
            self.students.remove(student)
            student.class_obj = None
            print(f"学员{student.name}已从{self.class_num}班级退学")
        else:
            print(f"学员{student.name}不在{self.class_num}班级中")

class Student:
    def __init__(self,name,age,degree):
        self.name = name
        self.age = age
        self.degree = degree
        self.class_obj = None  # 学员所报班级对象
        self.balance = 0  # 学员账户余额
    
    def pay_tuition(self):
        '''交学费（自动关联班级对应的课程价格）'''
        if self.class_obj:
            tuition = self.class_obj.course.price
            if self.balance >= tuition:
                self.balance -= tuition
                self.class_obj.school.account_balance +=tuition
                print(f"学员{self.name}已缴纳{self.class_obj.course.name}课程学费{tuition}元，剩余余额{self.balance}元")
            else:
                print(f"学员{self.name}余额不足，无法缴纳学费")
        else:
            print(f"学员{self.name}尚未报班，无法缴纳学费")
    
    def transfer_class(self,new_class):
        '''学员转班（去其他校区）'''
        if self.class_obj:
            old_class = self.class_obj
            old_class.drop_out(self)
            new_class.students.append(self)
            self.class_obj = new_class
            print(f"学员{self.name}已转至{new_class.school.name}校区的{new_class.class_num}班级")
        else:
            print(f"学员{self.name}未报名任何班级，无法转校")


class School:
    '''总部类'''
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.account_balance = 0  # 总部账户余额
        self.branches = []  # 下属分校列表
        self.staffs = []  # 总部员工列表
    
    def pay_rool(self,staff_num):
        '''发工资'''
        total_salary = sum([s.salary for s in self.staffs[:staff_num]])
        if self.account_balance >= total_salary:
            self.account_balance -= total_salary
            print(f"总部已给{staff_num}名员工发放工资，总金额{total_salary}元，剩余账户余额{self.account_balance}元")
        else:
            print("总部账户余额不足，无法发放工资")
    
    def count_staff_num(self):
        '''统计员工总数（总部和分部）'''
        total = len(self.staffs)
        for branch in self.branches:
            total += len(branch.staffs)
        print(f"总部+分校总员工数：{total}人")
        return total

    def count_stu_num(self):
        '''统计学生总数（总部和分部）'''
        total = 0 #总部没有学员
        for branch in self.branches:
            for cls in branch.classes:
                total += len(cls.students)
        print(f"总部+分校总学员数：{total}人")
        return total


    def new_staff_enrollment(self,staff):
        '''新员工注册（总部）'''
        self.staffs.append(staff)
        print(f"总部新增员工：{staff.name}（职位：{staff.position}）")

class BranchSchool:
    '''分校类'''
    def __init__(self,name,address,headquarter):
        self.name = name
        self.address = address
        self.headquarter = headquarter  # 所属总部
        self.account_balance = 0  # 分校账户余额
        self.staffs = []  # 分校员工列表
        self.classes = []  # 分校班级列表
        # 自动将分校加入总部的下属列表
        headquarter.branches.append(self)

    def add_class(self, cls):
        """分校新增班级"""
        self.classes.append(cls)
        print(f"{self.name}分校新增班级：{cls.class_num}（课程：{cls.course.name}）")

# --------------------测试--------------------
if __name__ == "__main__":
    # 1. 创建总部和多个分校（新增广州分校）
    headquarter = School("Python老司机总部", "北京市海淀区")
    branch_shanghai = BranchSchool("Python老司机上海分校", "上海市浦东新区", headquarter)
    branch_guangzhou = BranchSchool("Python老司机广州分校", "广州市天河区", headquarter)  # 新增广州分校

    # 2. 创建课程
    course_python = Course("Python面向对象", 2999, "包含类、继承、多态等核心知识点")
    course_java = Course("Java基础", 2499, "包含JVM、集合框架等内容")

    # 3. 创建不同分校的班级
    # 上海分校的班级
    class_python10_sh = Class("Python第10期（上海）", course_python, branch_shanghai)
    class_java5_sh = Class("Java第5期（上海）", course_java, branch_shanghai)
    branch_shanghai.add_class(class_python10_sh)
    branch_shanghai.add_class(class_java5_sh)

    # 广州分校的班级（新增）
    class_python8_gz = Class("Python第8期（广州）", course_python, branch_guangzhou)
    branch_guangzhou.add_class(class_python8_gz)

    # 4. 创建讲师并关联班级
    teacher_li = Teacher("李老师", 35, 15000, "教学部", branch_shanghai)
    teacher_li.teaching(class_python10_sh)
    class_python10_sh.create_teaching_record()
    teacher_wang = Teacher("王老师", 30, 13000, "教学部", branch_guangzhou)
    teacher_wang.teaching(class_java5_sh)
    class_java5_sh.create_teaching_record()


    # 5. 创建学员并报名上海分校的班级、交学费
    stu_zhou = Student("小周", 23, "硕士")
    stu_zhou.balance = 4000
    class_python10_sh.students.append(stu_zhou)
    stu_zhou.class_obj = class_python10_sh
    stu_zhou.pay_tuition()  # 先交上海班级的学费

    stu_wang = Student("小王", 25, "本科")
    stu_wang.balance = 2000
    class_java5_sh.students.append(stu_wang)
    stu_wang.class_obj = class_java5_sh
    stu_wang.pay_tuition()  # 交上海班级的学费

    # 6. 学员跨校区转班（从上海分校 → 广州分校）
    print("\n--- 开始跨校区转班 ---")
    stu_zhou.transfer_class(class_python8_gz)  # 传入广州分校的班级对象

    # 7. 总部统计数据（验证跨校区后学员归属）
    print("\n--- 统计全机构数据 ---")
    headquarter.count_staff_num()
    headquarter.count_stu_num()