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

> **Redacted 2026-07-05.** The original example here was a verbatim excerpt of a
> real statement (name, address, account number, real transactions) in a public
> repo. Replaced with a synthetic equivalent that preserves the layout the
> extractor targets. NOTE: the original text remains in git history until a
> history rewrite is done.

--- Page 1/3 ---
Page 1of 3
Delivery Method F1 R04
Branch Number Account Number Date DDA 00/00/XX FN
0000 0000 60000000000 2025/05/22 FNB PREMIER CURRENT ACCOUNT
MR J SOAP Universal Branch Code 250655
1 EXAMPLE STREET
SOMEWHERE
0000
Statement Period : 22 April 2025 to 22 May 2025
Statement Date : 22 May 2025
Statement Balances Bank Charges Interest Rate
Opening Balance 3,000.00Cr Service Fees 300.00Dr Credit Rate** Tiered
Closing Balance 16,000.00Cr Cash Deposit Fees 0.00 Debit Rate* 0.00%
Transactions in RAND (ZAR)
Accrued
Date Description Amount Balance Bank Charges
23 Apr Internet Pmt To Example Payee Ref-0000000000 1,120.00 1,880.00Cr 8.50
23 Apr POS Purchase Example Cafe 000000*0000 21 Apr 79.00 1,801.00Cr
24 Apr POS Purchase Example Grocer 000000*0000 20 Apr 556.80 1,244.20Cr

This data needs to be put in the csv like this:
Date, Description, Amount, Balance
23 Apr, Internet Pmt To Example Payee Ref-0000000000, 1,120.00, 1,880.00
23 Apr, POS Purchase Example Cafe 000000*0000 21 Apr, 79.00, 1,801.00
24 Apr, POS Purchase Example Grocer 000000*0000 20 Apr, 556.80, 1,244.20

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