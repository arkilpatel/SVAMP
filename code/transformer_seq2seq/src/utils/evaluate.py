from sympy import Eq, solve
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import pdb

"""
EXAMPLE:
prefix1 = '* n0 + + n1 n2 n3'
list_num1 = [13, 9, 10, 3] n0->13, n1->9, n2->10, n3->3
print(ans_evaluator(prefix1, list_num1))
answer: 286
"""

def format_eq(eq):
	fin_eq = ""
	ls = ['0','1','2','3','4','5','6','7','8','9','.']
	temp_num = ""
	flag = 0
	for i in eq:
		if flag > 0:
			fin_eq = fin_eq + i
			flag = flag-1
		elif i == 'n':
			flag = 6
			if fin_eq == "":
				fin_eq = fin_eq + i
			else:
				fin_eq = fin_eq + ' ' + i
		elif i in ls:
			temp_num = temp_num + i
		elif i == ' ':
			if temp_num == "":
				continue
			else:
				if fin_eq == "":
					fin_eq = fin_eq + temp_num
				else:
					fin_eq = fin_eq + ' ' + temp_num
			temp_num = ""
		else:
			if fin_eq == "":
				if temp_num == "":
					fin_eq = fin_eq + i
				else:
					fin_eq = fin_eq + temp_num + ' ' + i
			else:
				if temp_num == "":
					fin_eq = fin_eq + ' ' + i
				else:
					fin_eq = fin_eq + ' ' + temp_num + ' ' + i
			temp_num = ""
	if temp_num != "":
		fin_eq = fin_eq + ' ' + temp_num
	return fin_eq

def prefix_to_infix(prefix):
	operators = ['+', '-', '*', '/']
	stack = []
	elements = format_eq(prefix).split()
	for i in range(len(elements)-1, -1, -1):
		if elements[i] in operators and len(stack)>1:
			op1 = stack.pop(-1)
			op2 = stack.pop(-1)
			fin_operand = '(' + ' ' + op1 + ' ' + elements[i] + ' ' + op2 + ' ' + ')'
			stack.append(fin_operand)
		else:
			stack.append(elements[i])
	try:
		return stack[0]
	except:
		return ''

def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op

def back_align(eq, list_num):
	elements = eq.split()
	for i in range(len(elements)):
		if elements[i][0] == 'n':
			index = int(elements[i][6])
			try:
				number = str(list_num[index])
			except:
				return '-1000.112'
			elements[i] = number
	return stack_to_string(elements)    

def ans_evaluator(eq, list_num):
	#pdb.set_trace()
	infix = prefix_to_infix(eq)
	aligned = back_align(infix, list_num)
	try:
		final_ans = parse_expr(aligned, evaluate = True)
	except:
		final_ans = -1000.112
	return final_ans

def cal_score(outputs, nums, ans):
	corr = 0
	tot = 0
	disp_corr = []
	for i in range(len(outputs)):
		op = stack_to_string(outputs[i])
		num = nums[i].split()
		num = [float(nu) for nu in num]
		answer = ans[i].item()

		pred = ans_evaluator(op, num)

		if abs(pred - answer) <= 0.01:
			corr+=1
			tot+=1
			disp_corr.append(1)
		else:
			tot+=1
			disp_corr.append(0)

	return corr, tot, disp_corr

def get_infix_eq(outputs, nums):
	eqs = []
	for i in range(len(outputs)):
		op = stack_to_string(outputs[i])
		num = nums[i].split()
		num = [float(nu) for nu in num]

		infix = prefix_to_infix(op)
		eqs.append(infix)

	return eqs

