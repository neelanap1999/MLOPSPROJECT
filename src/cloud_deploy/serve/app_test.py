import requests
import json

health_check_url = 'http://127.0.0.1:8080/ping'

response_health = requests.get(health_check_url)
health_status = response_health.json()

print(health_status)


predict_url  = 'http://127.0.0.1:8080/predict'

data = {
   "instances": [
    {"loan_amnt":5000.0,"term":" 36 months","int_rate":18.99,"installment":183.26,"grade":"E","sub_grade":"E1","emp_title":"sales manager","emp_length":"10+ years","home_ownership":"MORTGAGE","annual_inc":40000.0,"verification_status":"Not Verified","issue_d":"2014-06-01","loan_status":"Fully Paid","purpose":"debt_consolidation","title":"Debt consolidation","dti":12.27,"earliest_cr_line":"Sep-2006","open_acc":14.0,"pub_rec":0.0,"revol_bal":4255.0,"revol_util":21.2,"total_acc":19.0,"initial_list_status":"f","application_type":"INDIVIDUAL","mort_acc":5.0,"pub_rec_bankruptcies":0.0,"address":"7622 Shannon Knoll\r\nNorth Johnfort, MA 30723"},
    {"loan_amnt":10625.0,"term":" 36 months","int_rate":16.99,"installment":378.76,"grade":"D","sub_grade":"D3","emp_title":"Shipping Department","emp_length":"10+ years","home_ownership":"RENT","annual_inc":33500.0,"verification_status":"Not Verified","issue_d":"2014-05-01","loan_status":"Charged Off","purpose":"debt_consolidation","title":"Debt consolidation","dti":20.1,"earliest_cr_line":"Feb-2006","open_acc":8.0,"pub_rec":0.0,"revol_bal":8739.0,"revol_util":69.0,"total_acc":9.0,"initial_list_status":"f","application_type":"INDIVIDUAL","mort_acc":0.0,"pub_rec_bankruptcies":0.0,"address":"506 Hernandez Lights Suite 385\r\nTroyhaven, MT 30723"},
    {"loan_amnt":15000.0,"term":" 36 months","int_rate":17.57,"installment":539.06,"grade":"D","sub_grade":"D4","emp_title":"city carrier","emp_length":"10+ years","home_ownership":"MORTGAGE","annual_inc":68000.0,"verification_status":"Verified","issue_d":"2014-06-01","loan_status":"Fully Paid","purpose":"credit_card","title":"Credit card refinancing","dti":22.44,"earliest_cr_line":"Feb-2004","open_acc":13.0,"pub_rec":0.0,"revol_bal":18230.0,"revol_util":50.7,"total_acc":25.0,"initial_list_status":"f","application_type":"INDIVIDUAL","mort_acc":2.0,"pub_rec_bankruptcies":0.0,"address":"107 Danielle Court Apt. 796\r\nLake Daniel, IL 22690"},{"loan_amnt":28000.0,"term":" 36 months","int_rate":12.99,"installment":943.3,"grade":"C","sub_grade":"C1","emp_title":"Business System Analyst - Oracle","emp_length":"10+ years","home_ownership":"RENT","annual_inc":84800.0,"verification_status":"Source Verified","issue_d":"2014-07-01","loan_status":"Fully Paid","purpose":"debt_consolidation","title":"Debt consolidation","dti":16.81,"earliest_cr_line":"May-1996","open_acc":11.0,"pub_rec":0.0,"revol_bal":25981.0,"revol_util":56.1,"total_acc":36.0,"initial_list_status":"f","application_type":"INDIVIDUAL","mort_acc":1.0,"pub_rec_bankruptcies":0.0,"address":"USNV Oconnor\r\nFPO AP 29597"}]
#[
#         {
#             "loan_amnt":5000.0,
#             "term":" 36 months",
#             "int_rate":18.99,
#             "installment":183.26,
#             "grade":"E",
#             "sub_grade":"E1",
#             "emp_title":"sales manager",
#             "emp_length":"10+ years",
#             "home_ownership":"MORTGAGE",
#             "annual_inc":40000.0,
#             "verification_status":"Not Verified",
#             "issue_d":"2014-06-01",
#             "loan_status":"Fully Paid",
#             "purpose":"debt_consolidation",
#             "title":"Debt consolidation",
#             "dti":12.27,
#             "earliest_cr_line":"Sep-2006",
#             "open_acc":14.0,"pub_rec":0.0,
#             "revol_bal":4255.0,
#             "revol_util":21.2,
#             "total_acc":19.0,
#             "initial_list_status":"f",
#             "application_type":"INDIVIDUAL",
#             "mort_acc":5.0,
#             "pub_rec_bankruptcies":0.0,
#             "address":"7622 Shannon Knoll\r\nNorth Johnfort, MA 30723"
#         }
#     ]
}

response_predict = requests.post(predict_url, json=data)
predict_status = response_predict.json()

print(predict_status)