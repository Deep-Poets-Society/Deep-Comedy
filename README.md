# Deep-Comedy

Syllabification of italian hendecasyllables and generation of new text in Dante's style with two transformers trained on Divina Comedia.
See our [report](https://github.com/Deep-Poets-Society/Deep-Comedy/blob/main/report.pdf).

## Syllabification of text [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-Poets-Society/Deep-Comedy/blob/main/syllabification.ipynb)

```
Original:	saver fu messo che se ’l vero è vero
Predicted:	|sa|ver |fu |mes|so |che $se ’l |ve|ro è |ve|ro
True:		|sa|ver |fu |mes|so |che $se ’l |ve|ro è |ve|ro

Original:	e sempre di mirar faceasi accesa
Predicted:	|e |sem|pre |di |mi|rar $fa|cea|sa|si ac|ce|sa
True:		|e |sem|pre |di |mi|rar $fa|cea|si ac|ce|sa

Original:	e bëatrice forse maggior cura
Predicted:	|e |bë|a|tri|ce |for|se $mag|gior |cru|ra
True:		|e |bë|a|tri|ce |for|se $mag|gior |cu|ra
```

![download](https://user-images.githubusercontent.com/31796254/134145791-0c11fb0d-327e-4ef6-80cb-81416fe91374.png)

## Generation of new text [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-Poets-Society/Deep-Comedy/blob/main/generation.ipynb)

```
seed:
 i’ fui colui che la ghisolabella
 condussi a far la voglia del marchese
 come che suoni la sconcia novella 

---generated:---

|ben |m’ ac|cor|si |ch’ el|li e|ra $da |ciel |mes|so
|ch’ io |vol|gea i |cad|di e $ri|ma|se e ’l |vol|to
|non |per |ve|der |non |ei $com’ |io |ti |mes|so

|co|sì |l’ ae|re |vi|cin $qui|vi |si |mi|ra
|qui|vi è il |gran |d’ o|gne $par|te u|dir |quin|ta
|co|sì |ri|spuo|se a |me $che |ti |ri|mi|ra

|e |l’ al|tro |dis|se |quel $che |tu |hai |guar|do
|ri|tro|ve|rai |co|me a $quel|l’ uom |ti|ra
|co|me an|cor |ti |sa|reb|be in $ma|ra|vi|glia
```

Checkpoints are stored on a [Gitlab repository](https://gitlab.com/sasso-effe/deep-comedy-checkpoints) since they are too big for GitHub.
