/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - musicreco
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`musicreco` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `musicreco`;

/*Table structure for table `userdetailes` */

DROP TABLE IF EXISTS `userdetailes`;

CREATE TABLE `userdetailes` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(200) default NULL,
  `address` varchar(200) default NULL,
  `lang` varchar(200) default NULL,
  `phone` varchar(200) default NULL,
  `email` varchar(200) default NULL,
  `password` varchar(200) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `userdetailes` */

insert  into `userdetailes`(`id`,`name`,`address`,`lang`,`phone`,`email`,`password`) values (2,'yash','india','hindi','9090909090','a','a'),(4,'sushant','india','hindi','9090909090','stawar59@gmail.com','123'),(5,'aarushi','india','hindi','9090909090','aarushi@gmail.com','1234'),(6,'piyush','India','hindi','9092939292','piyush@gmail.com','123456'),(7,'roshan','rsvvsvvvvvss','hindi','1234567890','yashsalvi1999@gmail.com','roshan');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
