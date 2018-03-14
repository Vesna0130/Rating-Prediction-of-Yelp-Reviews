library(ggplot2)
class=c(rep('acc',10), rep('val_acc',10))
Epoch=c(1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10)
value=c(0.4011,0.532,0.5671,0.5849,0.5986,0.6133,0.6237,0.6319,0.6418,0.6482,
        0.4903,0.5507,0.5664,0.5743,0.5785,0.5848,0.5853,0.5835,0.5828,0.5881)
tgg=data.frame(class,Epoch,value)
ggplot(tgg, aes(x=factor(Epoch), y=value, colour=class,group=class,shape=class,fill=class))+
  geom_line(size=.7)+geom_point(size=2)+
  xlab("Epoch")+ylab("value of accuracy and loss") +
  theme(panel.grid = element_blank()) +
  scale_colour_manual(values = c("#c49a1d", "#246aa9"))

class=c(rep('loss',10), rep('val_loss', 10))
Epoch=c(1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10)
value=c(1.3509,1.0748,0.9997,0.9543,0.9203,0.8935,0.869,0.8481,0.8281,0.8119,
        1.1686,1.0291,0.9927,0.9708,0.9657,0.9496,0.9518,0.9578,0.9515,0.9635)
tgg1=data.frame(class,Epoch,value)
ggplot(tgg1, aes(x=factor(Epoch), y=value, colour=class,group=class,shape=class,fill=class))+geom_line(size=.7)+geom_point(size=2)+xlab("Epoch")+ylab("value of valid accuracy and valid loss") +
theme(panel.grid = element_blank()) +
  scale_colour_manual(values = c("#c49a1d", "#246aa9"))

