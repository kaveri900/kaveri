CREATE DATABASE employee;
USE employee;
CREATE TABLE  employee (
emp_id int primary key,
emp_name varchar(50),
gender varchar(10),
age int,
department varchar(30),
destignation varchar(30),
salary decimal(10, 2),
City VARCHAR(30),
Manager_ID INT,
Joining_Date DATE
);

INSERT INTO Employee VALUES
(101,'Amit','Male',28,'IT','Developer',60000,'Bangalore',201,'2022-01-15'),
(102,'Sneha','Female',25,'HR','HR Executive',45000,'Mumbai',202,'2023-03-20'),
(103,'Rahul','Male',30,'IT','Developer',70000,'Bangalore',201,'2021-05-18'),
(104,'Priya','Female',27,'Finance','Analyst',55000,'Chennai',203,'2022-08-10'),
(105,'Karan','Male',35,'Sales','Manager',90000,'Delhi',204,'2020-02-25'),
(106,'Divya','Female',29,'IT','Tester',50000,'Hyderabad',201,'2022-09-15'),
(107,'Arjun','Male',31,'Finance','Analyst',65000,'Mumbai',203,'2021-11-05'),
(108,'Meena','Female',26,'Sales','Executive',40000,'Delhi',204,'2023-01-10'),
(109,'Rohit','Male',24,'HR','Recruiter',42000,'Bangalore',202,'2023-06-01'),
(110,'Anjali','Female',32,'IT','Team Lead',95000,'Hyderabad',201,'2019-12-20');

select * from employee;
select * from employee where salary >60000;

select distinct department
from employee;

select * from employee
order by salary DESC;

select * from employee
order by salary asc;

select * from employee
where emp_name like 'A%';

select * from employee
where salary between 50000 and 70000;

select * from employee
where city in('banglore', 'delhi');

select * from employee
where manager_id is null;

select department, count(*) as employee
from employee
group by department;

SELECT
COUNT(*) AS TotalEmployee,
SUM(Salary) AS TotalSalary,
AVG(Salary) AS AverageSalary,
MAX(Salary) AS HighestSalary,
MIN(Salary) AS LowestSalary
FROM Employee;


 
