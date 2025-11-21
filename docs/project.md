The goal of the project is to get insight into expenses by categorising / analysing my bank statements

## goal
the app should be able to:
1. Exctract table with transactions from the bank statement PDF
2. Create a csv with all transactions
3. an LLM should guess the category of the transcations based on the description and amount. The category needs to be based on a fixed ENUM
4. a model should be trainied based on the data in 3
5. future bank statements PDF should get a category on the trained model 

## technical details / guidelines
front-end: single page HTML, tailwind 
Back-end: python flask
Package manger and setup: python UV
Programming style: functional


## Details 1  & 2 extract data from pdf to csv
### here is an example of data from PDF
--- Page 1/3 ---
Page 1of 3
Delivery Method F1 R04
Branch Number Account Number Date DDA 07/94/CV/KM/KM/PA/P6/A6/LE/Y FN
NS/22/WV/DDA 07
2214 2214 62805977264 2025/05/22 FNB PREMIER CURRENT ACCOUNT
74698
XSTZFN0:62805977264
6 P O Box 12373
Mill Street, Post Office ,8010
Street Address Gardens
BBST74 031898 Thebe Hoskins House,Cnr Mill & Breda Sts
MR SANDER WIERSMA Universal Branch Code 250655
9 fnb.co.za
1 BOEKNHOUT MEWS STREET
Lost Cards 087-575-9406
MILKWOOD PARK SUNNYDALE
Account Enquiries 087-575-9404
NOORDHOEK Fraud 087-575-9444
7979 Relationship Manager Pw Service Suite
1
4 (087) 730-6000
Customer VAT Registration Number Not Provided FNB Premier Current Account : 62805977264
Bank VAT Registration Number 4210102051
Tax Invoice/Statement Number : 74
Statement Period : 22 April 2025 to 22 May 2025
Statement Date : 22 May 2025
Statement Balances Bank Charges Interest Rate
Opening Balance 3,544.96Cr Service Fees 302.02Dr Credit Rate** Tiered
Closing Balance 16,969.34Cr Cash Deposit Fees 0.00 Debit Rate* 0.00%
39.78Dr Cash Handling Fees 0.00
Total VAT (ZAR) 39.78Dr Other Fees 3.00Dr
Transactions in RAND (ZAR)
Accrued
Date Description Amount Balance Bank Charges
23 Apr Internet Pmt To Backabuddy Bo-Gcvr47V0Cbywda2Fo 1,120.00 2,424.96Cr 8.50
23 Apr POS Purchase Yoco *The Ice Caf 479012*1897 21 Apr 79.00 2,345.96Cr
23 Apr POS Purchase Yoco *Fish Hoek T 479012*1897 21 Apr 230.00 2,115.96Cr
24 Apr POS Purchase Cafe Roux 479012*1897 19 Apr 200.00 1,915.96Cr
24 Apr POS Purchase The Foodbarn Deli 479012*1897 19 Apr 300.00 1,615.96Cr
24 Apr POS Purchase Brass Bell Restaura 479012*1897 21 Apr 420.00 1,195.96Cr
24 Apr POS Purchase Spar Fish Hoek 479012*1897 20 Apr 556.80 639.16Cr

This data needs to be put in the csv like this:
Date, Description, Amount, Balance
23 Apr, Internet Pmt To Backabuddy Bo-Gcvr47V0Cbywda2Fo, 1,120.00,2,424.96
23 Apr, POS Purchase Yoco *The Ice Caf 479012*1897 21 Apr, 79.00, 2,345.96
23 Apr, POS Purchase Yoco *Fish Hoek T 479012*1897 21 Apr, 230.00, 2,115.96

## Details 3 use open ai API with structured output
The openai structured output API needs to be used to add the category.

Based on the request the following output needs to be created
Date: dd-mm-yyyy - 23-04-2025
Description: text
Amount: 1,1200.00
Balance: 1,1200.00
Category: 
	1.	Housing & Utilities - rent, mortgage, electricity, water, internet, cleaning services
	2.	Groceries & Household – supermarket purchases, home supplies, toiletries
	3.	Childcare & Education – daycare, school fees, uniforms, books, aftercare, activities
	4.	Transport – fuel, public transport, car payments, insurance, school transport
	5.	Health & Insurance – doctor visits, medication, medical aid, dental, health insurance
	6.	Food & Dining – restaurants, cafés, takeaways
	7.	Clothing & Personal Care – clothes, shoes, haircuts, personal grooming
	8.	Leisure & Entertainment – family outings, holidays, subscriptions, birthdays, toys
	9.	Financial & Miscellaneous – bank fees, savings, donations, memberships

See documentation in file: openai_structured_output.md