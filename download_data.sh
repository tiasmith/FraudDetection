#!/bin/bash

mkdir data
aws s3 cp s3://fraudulent-transactions/creditcard.csv data
mkdir output
