# syntax=docker/dockerfile:1

FROM php:7.2-apache

COPY ./PhpProjectTest /var/www/html

ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data

RUN chmod -R 775 /var/www/html
RUN chown -R root:www-data /var/www
RUN chmod u+rwx,g+rx,o+rx /var/www
RUN find /var/www -type d -exec chmod u+rwx,g+rx,o+rx {} +
RUN find /var/www -type f -exec chmod u+rw,g+rw,o+r {} +

RUN mkdir ./Python
ADD ./TFG_jchulilla_PEC3_TEST_ULM.py ./Python
ADD ./TFG_jchulilla_PEC3_TEST_LR.py ./Python
ADD ./TFG_jchulilla_PEC3_TEST_MNB.py ./Python
ADD ./learn.pkl ./Python
ADD ./tfidf_vectorizer.pkl ./Python
ADD ./modelLR.pkl ./Python
ADD ./modelMNB.pkl ./Python

COPY ./requirements.txt requirements.txt

RUN apt-get update
RUN apt-get -qq install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 80
CMD [ "/usr/sbin/apache2ctl", "-D", "FOREGROUND"]
