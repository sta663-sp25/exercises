FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y r-base
RUN Rscript -e "install.packages(c('tibble','stringi'))"

CMD ["R", "--vanilla"]
