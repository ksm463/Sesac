```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    

* clothes_dataset.zip 압축해제


```python
!pwd
```

    /content
    


```python
!mkdir clothes_dataset
```

    mkdir: cannot create directory ‘clothes_dataset’: File exists
    


```python
!unzip '/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/clothes_dataset.zip' -d ./clothes_dataset/
```

    [1;30;43m스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.[0m
      inflating: ./clothes_dataset/brown_shoes/29c9bf50db194174be8c8dafa3373f3eb337bb2d.jpg  
      inflating: ./clothes_dataset/brown_shoes/2b5552c967c7f406b1b780e9188cdbadcf4be6d1.jpg  
      inflating: ./clothes_dataset/brown_shoes/2b802f7d6e2b73ed86924aa7a8a8c65e0b9661ca.jpg  
      inflating: ./clothes_dataset/brown_shoes/2d202ed49ac15aeff75681b0d78c322c68852785.jpg  
      inflating: ./clothes_dataset/brown_shoes/2df99267923b05486744f5fe59ffb9dbf0dce133.jpg  
      inflating: ./clothes_dataset/brown_shoes/2e94cdf3e3020cad6ad91f01cf75264a270908c6.jpg  
      inflating: ./clothes_dataset/brown_shoes/2e9ec5a783dd8eb88af84a2fa01dd44ec0837e19.jpg  
      inflating: ./clothes_dataset/brown_shoes/2ee1746aacd7d31ab3a941934a11b9b34ae884bc.jpg  
      inflating: ./clothes_dataset/brown_shoes/2f098f41527b7ef3404c1df984f775d21b10df15.jpg  
      inflating: ./clothes_dataset/brown_shoes/2fbff880bb5d7c001b8993372107a9d06ac44a7f.jpg  
      inflating: ./clothes_dataset/brown_shoes/3034f61d1f09748763c3e1b0dadd915392b1278b.jpg  
      inflating: ./clothes_dataset/brown_shoes/30a1704559dec214b60e8bfad6826fdfadf1f87a.jpg  
      inflating: ./clothes_dataset/brown_shoes/30ef20bcb027c99409c81fd6127957502b0e693e.jpg  
      inflating: ./clothes_dataset/brown_shoes/312cf581fd4ec3678b8794f9f488aa1dad2f2908.jpg  
      inflating: ./clothes_dataset/brown_shoes/337a443bf71b424c5ff4bfa06bdfcc2a447f9535.jpg  
      inflating: ./clothes_dataset/brown_shoes/33a20d631a24cdd66b4a52d70029bd2037451a96.jpg  
      inflating: ./clothes_dataset/brown_shoes/343de6c46da4b0d867fed16827862cee5cb44fd5.jpg  
      inflating: ./clothes_dataset/brown_shoes/347cd771581ffc1c7fe7049e1823d5282361bc3f.jpg  
      inflating: ./clothes_dataset/brown_shoes/3550f40e92ae504615c57053b64221dafcfe178e.jpg  
      inflating: ./clothes_dataset/brown_shoes/3568f3e842d58733319fdd038dbb5cf50d63f7b3.jpg  
      inflating: ./clothes_dataset/brown_shoes/35c9001c72e8db467dcb8956e421aa5aca756da6.jpg  
      inflating: ./clothes_dataset/brown_shoes/36e76c0fea3e36e4495eb1652b2d287c698a7878.jpg  
      inflating: ./clothes_dataset/brown_shoes/3762df64de73b98c9690f096fe549d55940c6644.jpg  
      inflating: ./clothes_dataset/brown_shoes/3af2ed6a769a0aaae722664eff72b61d96252083.jpg  
      inflating: ./clothes_dataset/brown_shoes/3bf9ef4b2e13f7ab04d0fb4bdedba20c5fb39757.jpg  
      inflating: ./clothes_dataset/brown_shoes/3c6ec6140e00aad809d477bfa725c3e9412f397f.jpg  
      inflating: ./clothes_dataset/brown_shoes/3d1520ae86d2ad5b5881a25304d747cb638292d0.jpg  
      inflating: ./clothes_dataset/brown_shoes/3d420ebb7309b44fda29e0d5d68bddb60704aef5.jpg  
      inflating: ./clothes_dataset/brown_shoes/3e443c30dc7d004b2e44ac93b61304a1048a87ec.jpg  
      inflating: ./clothes_dataset/brown_shoes/3e789c6e7dfbf8a522643c6ba14a6539a95c25a2.jpg  
      inflating: ./clothes_dataset/brown_shoes/3e925a23e385486100533add5cf65f5507a1d2cd.jpg  
      inflating: ./clothes_dataset/brown_shoes/3ece522e8f055e7d909cff8fee42b875ecd1850a.jpg  
      inflating: ./clothes_dataset/brown_shoes/3f0cbcd84288d69267f4314a0386a23f57480a74.jpg  
      inflating: ./clothes_dataset/brown_shoes/3f7de9d7d587849ee58c338534cfb50aceb80ffe.jpg  
      inflating: ./clothes_dataset/brown_shoes/3f90c7e3251046e11a146f3766e3807f31f03320.jpg  
      inflating: ./clothes_dataset/brown_shoes/3fe1d6a6302dc6e82296f16c01cce067c86cd5df.jpg  
      inflating: ./clothes_dataset/brown_shoes/40269db7ce3ee02ee673e60bfa7cdd85b5572a94.jpg  
      inflating: ./clothes_dataset/brown_shoes/40e791a0faa68396ed042b113eaa66c4361845a9.jpg  
      inflating: ./clothes_dataset/brown_shoes/415657758d5048fded9dea5621ceead6cd474460.jpg  
      inflating: ./clothes_dataset/brown_shoes/416e19a033bd07650774fa4878c156de09f1829d.jpg  
      inflating: ./clothes_dataset/brown_shoes/42482e0532f2e04d49c92a4dcb4b03d95a91a1a9.jpg  
      inflating: ./clothes_dataset/brown_shoes/427d14778334d46120ed1ece2e3341d06ed5c503.jpg  
      inflating: ./clothes_dataset/brown_shoes/438b5365f1514778e18774bc7b5fea0a90a12784.jpg  
      inflating: ./clothes_dataset/brown_shoes/45185031e9c29825483daa812a68aef4e2ec1810.jpg  
      inflating: ./clothes_dataset/brown_shoes/4564016e4916528c577d5a52d5042bb53181ff72.jpg  
      inflating: ./clothes_dataset/brown_shoes/460cb15f60ba3d1279d4241587b194bbd2076962.jpg  
      inflating: ./clothes_dataset/brown_shoes/4656fc8f0cc145e20f0b01864e37ed3a9de5e40f.jpg  
      inflating: ./clothes_dataset/brown_shoes/4670f358e172a1b929c7d361fde6b69cca254441.jpg  
      inflating: ./clothes_dataset/brown_shoes/4687a9ac604e15a3fa842070a3897820f14a0a05.jpg  
      inflating: ./clothes_dataset/brown_shoes/47a59c3cb0192819a18d2f45606a238e5adfa998.jpg  
      inflating: ./clothes_dataset/brown_shoes/48d02ef24209465c0422b9ccec1fc5d0c064520d.jpg  
      inflating: ./clothes_dataset/brown_shoes/48d64af9497687986204b2e5170830eb9d5643ef.jpg  
      inflating: ./clothes_dataset/brown_shoes/48e16a23f8b591d65c449a931e28d283b5f85223.jpg  
      inflating: ./clothes_dataset/brown_shoes/48f531ba8791a1ff13c9247f6b499cacb1772f78.jpg  
      inflating: ./clothes_dataset/brown_shoes/493939cfca7a9ee9e75c00228ed28bedfbcb9e76.jpg  
      inflating: ./clothes_dataset/brown_shoes/4959022e6f9b2fd534305eb6191e5057daa6e43e.jpg  
      inflating: ./clothes_dataset/brown_shoes/4a427e856a309f20f41838f540d8028329e4e562.jpg  
      inflating: ./clothes_dataset/brown_shoes/4a7ffd817e2c2ec3294cf2a9df9fe5132e288b2d.jpg  
      inflating: ./clothes_dataset/brown_shoes/4c5493aa7491110bfb6c8dba1a120890bf93bfd4.jpg  
      inflating: ./clothes_dataset/brown_shoes/4d37298a53c2d161017f36ef7d52f4f24afd5020.jpg  
      inflating: ./clothes_dataset/brown_shoes/4de315c7bbbadbd61e418268a9b0466e830ae0b7.jpg  
      inflating: ./clothes_dataset/brown_shoes/4dfe6e446b08970ed82844ca5f5799055a99ad74.jpg  
      inflating: ./clothes_dataset/brown_shoes/4f0bf678666cbbd7d22f47bcdd10b9cb0b375c7f.jpg  
      inflating: ./clothes_dataset/brown_shoes/506513af4d0f41b80f38cf6c839b5f7380eacafc.jpg  
      inflating: ./clothes_dataset/brown_shoes/50922eeb7b4e3ea30a47b5641efc855f74133989.jpg  
      inflating: ./clothes_dataset/brown_shoes/52adbfc96b691177e0ce47f95c2c09114f578414.jpg  
      inflating: ./clothes_dataset/brown_shoes/5378e492ce5c8d3b3c53c7b9e0f2ddf74805ea15.jpg  
      inflating: ./clothes_dataset/brown_shoes/547f405b084ec9ef83548f969bb7b356471d34a7.jpg  
      inflating: ./clothes_dataset/brown_shoes/55492a0b0716772258af772da1274b91a0923188.jpg  
      inflating: ./clothes_dataset/brown_shoes/55f96500ccbcef99608ca079f6f8d78e53d188e7.jpg  
      inflating: ./clothes_dataset/brown_shoes/57dc8618138ff5d977d376a33e211be649f67b57.jpg  
      inflating: ./clothes_dataset/brown_shoes/58def8dff9f74d582961e979a838252dc34a856e.jpg  
      inflating: ./clothes_dataset/brown_shoes/599ea8e792121c68996f11a19223ab1687c9b7b3.jpg  
      inflating: ./clothes_dataset/brown_shoes/5a12f954e7dd973105b7edca84f0d52a75e1548d.jpg  
      inflating: ./clothes_dataset/brown_shoes/5a30e3fa4398c6f9d47aa47b5c2e33e64ee54c13.jpg  
      inflating: ./clothes_dataset/brown_shoes/5a9f9ae5e5ea873156973d29dfeafd5444c0cba7.jpg  
      inflating: ./clothes_dataset/brown_shoes/5adda4c2fd3c479cc702507c45cb28c43cf099b0.jpg  
      inflating: ./clothes_dataset/brown_shoes/5c3216947dd471d332cf6bcabb4a1b972eb92148.jpg  
      inflating: ./clothes_dataset/brown_shoes/5d2a3f4621dd23040a6705f1af15ff33fcc16f3c.jpg  
      inflating: ./clothes_dataset/brown_shoes/5dfed9f3a5f594348adb9d729ab34a30251188c1.jpg  
      inflating: ./clothes_dataset/brown_shoes/5e8486ad9e91e94e8aaf7a64b8b0d9701fc156de.jpg  
      inflating: ./clothes_dataset/brown_shoes/5ec30da2fbb9094976d059d8737f47e00391be23.jpg  
      inflating: ./clothes_dataset/brown_shoes/61af6c2a96c87550662b737f8b2a6d06dc83d901.jpg  
      inflating: ./clothes_dataset/brown_shoes/61ebdadcca607ce2e9a0e870ef6d58900be6010c.jpg  
      inflating: ./clothes_dataset/brown_shoes/624da1c6b13ce6629c0ddb0e6c861df7268c8cb4.jpg  
      inflating: ./clothes_dataset/brown_shoes/62c3ab83fd6566b9567b5076c73430616fd2ec0b.jpg  
      inflating: ./clothes_dataset/brown_shoes/64810300542bbb79f83b54ce414e0c9a90d543f6.jpg  
      inflating: ./clothes_dataset/brown_shoes/6536f1808210fdf51acda12cf4fb2ef9040e71f7.jpg  
      inflating: ./clothes_dataset/brown_shoes/659dae8a8fdfb79b2d3e32eac74f658b939883b5.jpg  
      inflating: ./clothes_dataset/brown_shoes/662074cfd841b891db882f0aba5ddc6446ea2f5f.jpg  
      inflating: ./clothes_dataset/brown_shoes/67534e3d16b583cdcbc2fc30383d396ab83a4f63.jpg  
      inflating: ./clothes_dataset/brown_shoes/6855f2ea37a45fbd7c2e534c57434082c19212aa.jpg  
      inflating: ./clothes_dataset/brown_shoes/69309747c0f6657ec817f8f2643494a4c52eddba.jpg  
      inflating: ./clothes_dataset/brown_shoes/69899ecfcd933176b31f68b5c4c967940364d305.jpg  
      inflating: ./clothes_dataset/brown_shoes/6a27b89813401957297199071040b2c8f5f0bcfd.jpg  
      inflating: ./clothes_dataset/brown_shoes/6a7faa3b248c45f6c2d7bb8f7db92464db95f44c.jpg  
      inflating: ./clothes_dataset/brown_shoes/6abc73bce3d3ab418dde2f9d1dc190b5ce1fa93d.jpg  
      inflating: ./clothes_dataset/brown_shoes/6abfad77024d0c268ea00023d63e5ea1df69759e.jpg  
      inflating: ./clothes_dataset/brown_shoes/6ac2fbf0fe69081231d4d70d1dba5211a0ac9914.jpg  
      inflating: ./clothes_dataset/brown_shoes/6b1247cb0d451032b44ba15e2ff6bdc3df3be201.jpg  
      inflating: ./clothes_dataset/brown_shoes/6d1ec50819d631cfe1dc2be3c169eaf247499bc0.jpg  
      inflating: ./clothes_dataset/brown_shoes/6dccc507a1748594e65071d5c25530364e73589f.jpg  
      inflating: ./clothes_dataset/brown_shoes/6df24807f45977759bde8ee6ccea7ed0cb687a60.jpg  
      inflating: ./clothes_dataset/brown_shoes/6e25e11ca8a8d925497b7947158ecb6bb5251f19.jpg  
      inflating: ./clothes_dataset/brown_shoes/6e8c95016ca11667ab70872feea43ebd948e729a.jpg  
      inflating: ./clothes_dataset/brown_shoes/6f2a633393dc47904e3f2e99c933edd67895bcad.jpg  
      inflating: ./clothes_dataset/brown_shoes/702b62a479feea03ee4eebbd47789e3b9f4ea77c.jpg  
      inflating: ./clothes_dataset/brown_shoes/7122d9c1f3157251edfb83da61990bd6fa8dacb4.jpg  
      inflating: ./clothes_dataset/brown_shoes/71579d49ff2fd9a85ee971c6b80076f4d555d890.jpg  
      inflating: ./clothes_dataset/brown_shoes/728810c9fcc0df9fc0f5810a44cef3426e47f071.jpg  
      inflating: ./clothes_dataset/brown_shoes/72a428028ca1f765e4ca98d544a9348f06ade6d3.jpg  
      inflating: ./clothes_dataset/brown_shoes/7346181875f4d5378e54e40e0410d8274de55771.jpg  
      inflating: ./clothes_dataset/brown_shoes/73462f431a026aaf21ee5f03bcfa960b3569e397.jpg  
      inflating: ./clothes_dataset/brown_shoes/73d0cc4a5b9bb8a099ded310f4b2ad8661b4ce76.jpg  
      inflating: ./clothes_dataset/brown_shoes/741197fef625ca11ccc0bbda9c28ad25dc723ad8.jpg  
      inflating: ./clothes_dataset/brown_shoes/744258c4a03daf6a728e3aa080c72ba8919305f6.jpg  
      inflating: ./clothes_dataset/brown_shoes/75e82a13417edad24d3b6c63377a6a6f75e2b487.jpg  
      inflating: ./clothes_dataset/brown_shoes/766d1e6b77c69d3c72573f0627955d6a59fad49e.jpg  
      inflating: ./clothes_dataset/brown_shoes/76abf4f9f805e73ee9f93e22978991161b91a5ec.jpg  
      inflating: ./clothes_dataset/brown_shoes/76e41fd58111589167f5298ea0601b965548408f.jpg  
      inflating: ./clothes_dataset/brown_shoes/77731148cef143edb07c1f46b2358a47eee96bd8.jpg  
      inflating: ./clothes_dataset/brown_shoes/7825bd673921a7ae75fe0877d8af922d1e388f4d.jpg  
      inflating: ./clothes_dataset/brown_shoes/790a228993237e0bfea00766497d6ff9a1939ba4.jpg  
      inflating: ./clothes_dataset/brown_shoes/7941677a421861e9607d6dfbcbc474e1215cb3f7.jpg  
      inflating: ./clothes_dataset/brown_shoes/7991587c8c3b49482a4f1f67e2a51711eed7e6c7.jpg  
      inflating: ./clothes_dataset/brown_shoes/79a9b88e58f96cb97ef696c185a65e29fd9cf2d9.jpg  
      inflating: ./clothes_dataset/brown_shoes/7bbaa1e2527ea041675de426d39acc3101a81088.jpg  
      inflating: ./clothes_dataset/brown_shoes/7c53337d191669e21830d1d52ab08c565e998d9e.jpg  
      inflating: ./clothes_dataset/brown_shoes/7c79d2d1d3eabe4796daaf5ebfffb067d7f704aa.jpg  
      inflating: ./clothes_dataset/brown_shoes/7dfa6440cdeb9f22584f5dbf06cae3d576ec579f.jpg  
      inflating: ./clothes_dataset/brown_shoes/7e90e664359e38ddd83fa68f79c652a3c78a80f6.jpg  
      inflating: ./clothes_dataset/brown_shoes/7f10140f647e81c2c8a3365fcadb7697f97e9ef2.jpg  
      inflating: ./clothes_dataset/brown_shoes/7f7d2405b5613dfd134f9bb311ce2e3bfd4b5799.jpg  
      inflating: ./clothes_dataset/brown_shoes/7fc3a06cee107e4b2c51089a38632138545fb4f4.jpg  
      inflating: ./clothes_dataset/brown_shoes/803ff782af49d69504433293e5a898bb3be28261.jpg  
      inflating: ./clothes_dataset/brown_shoes/81eaf6b8fe070d765ec2f45ea59c48747976331e.jpg  
      inflating: ./clothes_dataset/brown_shoes/8250a1b8e38c20c9ce0ce53bd76e2e1ecc14403c.jpg  
      inflating: ./clothes_dataset/brown_shoes/8269aa574449229966fdc99abfc31411db89f227.jpg  
      inflating: ./clothes_dataset/brown_shoes/827991b7d50544d3aa6ec0421c28b3b0f9460252.jpg  
      inflating: ./clothes_dataset/brown_shoes/829f2b598d75d97ed1e866737fa683f35d688b4c.jpg  
      inflating: ./clothes_dataset/brown_shoes/859fb359c8f97645747c99860c03aea631457e85.jpg  
      inflating: ./clothes_dataset/brown_shoes/86fab234f009b4b28ad905d5401248f4ee687494.jpg  
      inflating: ./clothes_dataset/brown_shoes/876b023b3bca72f9190f8d3563e34878204e61cd.jpg  
      inflating: ./clothes_dataset/brown_shoes/87a18ed5217f4b876f51ad6d12b8d6944e1a3b9c.jpg  
      inflating: ./clothes_dataset/brown_shoes/882aa3c8c07ed59c6bdcd66813cfc76d10a03e86.jpg  
      inflating: ./clothes_dataset/brown_shoes/8852347d558a7fd641780dc1ef72d1eff3a85b25.jpg  
      inflating: ./clothes_dataset/brown_shoes/8882566b93b605e5a8b2a9023aa0202555807aaa.jpg  
      inflating: ./clothes_dataset/brown_shoes/889e26d4b2664e87d066604200c8980de967a899.jpg  
      inflating: ./clothes_dataset/brown_shoes/88a70ed0e2f14e5cfe2b96d1f7c58b84648f220b.jpg  
      inflating: ./clothes_dataset/brown_shoes/8909911aeb29ce3472e42ef922850082aaeac774.jpg  
      inflating: ./clothes_dataset/brown_shoes/893dc6d770d1ab525317bd7e6553225d5e225df4.jpg  
      inflating: ./clothes_dataset/brown_shoes/8941acf56be5bbd53a4f1802147db536689d5fc4.jpg  
      inflating: ./clothes_dataset/brown_shoes/8a0476fdf5aa576569501b8a9f96d3a358bef8e0.jpg  
      inflating: ./clothes_dataset/brown_shoes/8aaeee1d81e9832a0b5782c3a37f70bcefad7aac.jpg  
      inflating: ./clothes_dataset/brown_shoes/8b839bbf944989f989a0f004a9b18b2ff5da94a2.jpg  
      inflating: ./clothes_dataset/brown_shoes/8be6c362f8e913e510995ba7d7db9165e917ae88.jpg  
      inflating: ./clothes_dataset/brown_shoes/8c5f569d99bc6f1b20d3c90558730b5c3ce40233.jpg  
      inflating: ./clothes_dataset/brown_shoes/8cc1140e7a87c54c0e83d693eb86b151fcc0335f.jpg  
      inflating: ./clothes_dataset/brown_shoes/8d146a29915f8d96f1ffa84c8e328676de87a6fd.jpg  
      inflating: ./clothes_dataset/brown_shoes/8d48d2814cad3d6f37dbe11857a87552d1a149ba.jpg  
      inflating: ./clothes_dataset/brown_shoes/8de7b22af5b7542de3e4627c3a192f03f78992cf.jpg  
      inflating: ./clothes_dataset/brown_shoes/8e97a4ee1e4ce175b00e3856ce7c011475dbde29.jpg  
      inflating: ./clothes_dataset/brown_shoes/8efd2437a9c4e39625f6f5caa60ad928d77e9f70.jpg  
      inflating: ./clothes_dataset/brown_shoes/8f785719f0516d5d2df2987d2540e0e96682298b.jpg  
      inflating: ./clothes_dataset/brown_shoes/8fe3df92cbbb1055a8e5652b471f4422c61f3167.jpg  
      inflating: ./clothes_dataset/brown_shoes/90f883c59d981e7c8cb1103bc8fa4faa1612c249.jpg  
      inflating: ./clothes_dataset/brown_shoes/918c74c8db2cebc53472c87b85de2493ed1e7497.jpg  
      inflating: ./clothes_dataset/brown_shoes/9198bfcdf03d5f86010d6cec18852edd06b5137f.jpg  
      inflating: ./clothes_dataset/brown_shoes/92308eca28ea3e8e9b5f8b7b3b71f9ab6b50badf.jpg  
      inflating: ./clothes_dataset/brown_shoes/935300860fcc9600fce886cc706ae57244775de1.jpg  
      inflating: ./clothes_dataset/brown_shoes/9396e531be83d5f74b646b77ee0aaeb7c2c4d8af.jpg  
      inflating: ./clothes_dataset/brown_shoes/93ac16facb20c7490eec616fab4460e76041c2a3.jpg  
      inflating: ./clothes_dataset/brown_shoes/93f39723584e53e7552b34dd99ffc54571577827.jpg  
      inflating: ./clothes_dataset/brown_shoes/940ad5abc70974ff35bc9fbf89ec53e6b15ad3d2.jpg  
      inflating: ./clothes_dataset/brown_shoes/9588ea571102dc1316107ff179ad57130f7a0919.jpg  
      inflating: ./clothes_dataset/brown_shoes/96227a0d50101a39e63afef4823f1b102a177e1c.jpg  
      inflating: ./clothes_dataset/brown_shoes/962ffb9be884178eba35b03245e2b09a9ad8f658.jpg  
      inflating: ./clothes_dataset/brown_shoes/96911a5108e81345ddfbfc74e25fb68fa41b0e98.jpg  
      inflating: ./clothes_dataset/brown_shoes/97033f3412bf59ba79a58fa51231c165e65102a3.jpg  
      inflating: ./clothes_dataset/brown_shoes/97b17caf0fc03489d66fbeaaf82ef2cb97cf0971.jpg  
      inflating: ./clothes_dataset/brown_shoes/983c63b75302f04db42db67a0a5f54492084a346.jpg  
      inflating: ./clothes_dataset/brown_shoes/9887b125603c894baf1cc0a6a6ea38834c0f7e95.jpg  
      inflating: ./clothes_dataset/brown_shoes/98d3c3df65632f02b3dbda640bc433cb301eaec5.jpg  
      inflating: ./clothes_dataset/brown_shoes/9af64477da026d87b219bf79db9197c7884d7085.jpg  
      inflating: ./clothes_dataset/brown_shoes/9ba388bd01e7cee6e4d138acddd6daf0142a0737.jpg  
      inflating: ./clothes_dataset/brown_shoes/9c6b238be1f5a2ae2abd90e13fad025aeade5495.jpg  
      inflating: ./clothes_dataset/brown_shoes/9cae35d14e83542603fb330fe5e9d37c0100bcf5.jpg  
      inflating: ./clothes_dataset/brown_shoes/9d09fa6897a4b48bd8def216b58418b8ab2eba8f.jpg  
      inflating: ./clothes_dataset/brown_shoes/9d6ca28744d6c5c5d15877058384ef4fbf625031.jpg  
      inflating: ./clothes_dataset/brown_shoes/9ddeccf3ccfb85c482d36ad1e6b8b2817f835b8a.jpg  
      inflating: ./clothes_dataset/brown_shoes/9e03ebf011153a0a3c05248a7cbfd7dd37012b68.jpg  
      inflating: ./clothes_dataset/brown_shoes/9e05f270a8cb9be9a22a0f69ab8e4a4ad34162f4.jpg  
      inflating: ./clothes_dataset/brown_shoes/9ed2b958ca95282a2277d4f56b7adbe349f1ad42.jpg  
      inflating: ./clothes_dataset/brown_shoes/9f9aee263bc3d51ae4ada695227e97611743e16c.jpg  
      inflating: ./clothes_dataset/brown_shoes/9fac2dcdaa383eee0a9cc01d5e73ebc784a7f049.jpg  
      inflating: ./clothes_dataset/brown_shoes/a06766f745b2d14bf4c33d9b671efbf782e65733.jpg  
      inflating: ./clothes_dataset/brown_shoes/a08c54c2b6f7850c31f0e971d6d54e670b1cf360.jpg  
      inflating: ./clothes_dataset/brown_shoes/a093a8df5ffc76ce562d6361092f8538bad35085.jpg  
      inflating: ./clothes_dataset/brown_shoes/a1d01aba738833c0e4ac7d5bfcfa43078e7c2dee.jpg  
      inflating: ./clothes_dataset/brown_shoes/a280387c328ec6e4888deb3ec83b83c32076723d.jpg  
      inflating: ./clothes_dataset/brown_shoes/a2c7a75f8cadaadd33b5e010dd7d25775367fe0c.jpg  
      inflating: ./clothes_dataset/brown_shoes/a315b41213ff509ef70c6f1579916925972b59e0.jpg  
      inflating: ./clothes_dataset/brown_shoes/a3fe4a1a9399b0457c7ac8d3490dd3c7dea6969a.jpg  
      inflating: ./clothes_dataset/brown_shoes/a486f244acf637c6317d68d2900004b46a1b6cf4.jpg  
      inflating: ./clothes_dataset/brown_shoes/a4baa30f9740a942ecc49a541277b787208f34df.jpg  
      inflating: ./clothes_dataset/brown_shoes/a59cc678bb6f2b0dcb8701d18cd439cbcdd059c4.jpg  
      inflating: ./clothes_dataset/brown_shoes/a87d5d0f6032f8050fa279295cf08538d3e2d2ba.jpg  
      inflating: ./clothes_dataset/brown_shoes/a8c952446e887c47893c727c53ceb5429ac2df4a.jpg  
      inflating: ./clothes_dataset/brown_shoes/a97bb5f10aba2d8657976633aed3390690cff3a1.jpg  
      inflating: ./clothes_dataset/brown_shoes/a99a1fb14076462948437212b26b8e460cc961a6.jpg  
      inflating: ./clothes_dataset/brown_shoes/a9df039cedaa204957f04ed80d19c549ae3a569e.jpg  
      inflating: ./clothes_dataset/brown_shoes/aaa9d123537a2b53410ed1bfaa09629a4581785d.jpg  
      inflating: ./clothes_dataset/brown_shoes/aab98416d601ea6ef5dc34e498e03a639de8960a.jpg  
      inflating: ./clothes_dataset/brown_shoes/ab62deb505859b1d10ef69dd3300f04228a57b02.jpg  
      inflating: ./clothes_dataset/brown_shoes/ab810fffcd60cb602a6afdbdd897817a81c4ff9f.jpg  
      inflating: ./clothes_dataset/brown_shoes/ace5934ad33152e2d84772be22c7577f065da273.jpg  
      inflating: ./clothes_dataset/brown_shoes/ad211be41d7493da428c0352860b420c8218f102.jpg  
      inflating: ./clothes_dataset/brown_shoes/ada46af2ab0aea0647d03d2a2a024ada6d539d77.jpg  
      inflating: ./clothes_dataset/brown_shoes/ae57c6c934c699c78dfe05e3756cb82192e20ffa.jpg  
      inflating: ./clothes_dataset/brown_shoes/aeb7c2cc7519123acf800504bcaea0fe37d69ed5.jpg  
      inflating: ./clothes_dataset/brown_shoes/af57d88e05277f69d45ba1a1958c51a5f3cd16c3.jpg  
      inflating: ./clothes_dataset/brown_shoes/af812b2cd21604fcb7922df2ad766bb5a765e3ec.jpg  
      inflating: ./clothes_dataset/brown_shoes/afde862a31dd64dc2841a844fdf1ed74503f449b.jpg  
      inflating: ./clothes_dataset/brown_shoes/b0917ad11e93f5294f0f77a2a75f0bc118738d72.jpg  
      inflating: ./clothes_dataset/brown_shoes/b0c87332165f0aa523314e07b4d2a0c074dc1b31.jpg  
      inflating: ./clothes_dataset/brown_shoes/b0cf0f1936103ea01c1d9a9add625bfa5ba6ea5d.jpg  
      inflating: ./clothes_dataset/brown_shoes/b0db04bcf9006127f388dc70e2c0a8873e129348.jpg  
      inflating: ./clothes_dataset/brown_shoes/b11db313a13d8f1b1cdf7bc46646236a50117042.jpg  
      inflating: ./clothes_dataset/brown_shoes/b155fa01576e525ebbe811d758e76a72a6d5d6f2.jpg  
      inflating: ./clothes_dataset/brown_shoes/b1b869b139fb796bbe253e824e081cd93561a471.jpg  
      inflating: ./clothes_dataset/brown_shoes/b2030f7579f327cae128d5d12d00b461a3241e50.jpg  
      inflating: ./clothes_dataset/brown_shoes/b23011929b584f691b1cd0ff808dbb3d2243104b.jpg  
      inflating: ./clothes_dataset/brown_shoes/b3bc7d622761b66a74cce0b886edee7c1709e9c3.jpg  
      inflating: ./clothes_dataset/brown_shoes/b445889b526702349bc4e960f050a2d85cc59967.jpg  
      inflating: ./clothes_dataset/brown_shoes/b4759c1d7807ab19777a320ff2104c4d2b374b6c.jpg  
      inflating: ./clothes_dataset/brown_shoes/b488fddd8132020cc9f879effe853249c5daef34.jpg  
      inflating: ./clothes_dataset/brown_shoes/b4c8448ae7cf0a3d3289c67a96566c9f6da6ecd8.jpg  
      inflating: ./clothes_dataset/brown_shoes/b4f69dfaebaae775e5fe4c638456820227617df9.jpg  
      inflating: ./clothes_dataset/brown_shoes/b5afb0bef1eda2fc02b9fa91875cdf65ae4da546.jpg  
      inflating: ./clothes_dataset/brown_shoes/b6505611d8fcdd990a55d3228ac9e8e151456106.jpg  
      inflating: ./clothes_dataset/brown_shoes/b6a236ab081feefcb08c431346f0d6a2112ebddd.jpg  
      inflating: ./clothes_dataset/brown_shoes/b6f917d00ec5c332d643e71842197814512c7b61.jpg  
      inflating: ./clothes_dataset/brown_shoes/b75809a588fe0976ccf614e90d90bed89b4d9a1b.jpg  
      inflating: ./clothes_dataset/brown_shoes/b7b11d7fd0554a7b67d0feafbe1dc95ce247b651.jpg  
      inflating: ./clothes_dataset/brown_shoes/b820c42d1c74f24323aedccf68ac795012d01722.jpg  
      inflating: ./clothes_dataset/brown_shoes/b845c0e725acc3570e89b66be66296c9c9e1c37b.jpg  
      inflating: ./clothes_dataset/brown_shoes/b8b87bfe0f17c9bf9d692ce1c26980df9b23b6e2.jpg  
      inflating: ./clothes_dataset/brown_shoes/b8f29ccb28204a6d545824cb07f2fa56388cdf67.jpg  
      inflating: ./clothes_dataset/brown_shoes/b9fee6c29b0846f4ea6a4316777965b6c1e07134.jpg  
      inflating: ./clothes_dataset/brown_shoes/ba1d0e179f3fbffd82e689e83b56b1bcc57fc469.jpg  
      inflating: ./clothes_dataset/brown_shoes/ba3f325de776a20b84692d9fdbcec7e7cdd87733.jpg  
      inflating: ./clothes_dataset/brown_shoes/badb8e306be7d628a3f125eefc823ef3ddc2deb1.jpg  
      inflating: ./clothes_dataset/brown_shoes/bb5b7f6a888470964cbc41b6563e813326fe56bf.jpg  
      inflating: ./clothes_dataset/brown_shoes/bb5cd7bc18f1eda6d4c236ee95661f6c21fc8573.jpg  
      inflating: ./clothes_dataset/brown_shoes/bb7ce0dadb08336e8be1f4589205196e9b3653c1.jpg  
      inflating: ./clothes_dataset/brown_shoes/bc32f48b7686a13b38cd898508dde4ce6f940841.jpg  
      inflating: ./clothes_dataset/brown_shoes/be15e427544dbfa6691b112841f45ebd9b5a06e5.jpg  
      inflating: ./clothes_dataset/brown_shoes/beff93f35c1bb5bad668af1a76481005236dbcfd.jpg  
      inflating: ./clothes_dataset/brown_shoes/bf00422cd6c43d8c63d17b04ab309c82ebd3513e.jpg  
      inflating: ./clothes_dataset/brown_shoes/c016c869a1f4869deefb70da6fbdbd24eb5b7c4b.jpg  
      inflating: ./clothes_dataset/brown_shoes/c0d350dfd5f662247d6fac7238197f2fc2e8aa37.jpg  
      inflating: ./clothes_dataset/brown_shoes/c183d708045a8238779472a5f0111d0d07bb2d79.jpg  
      inflating: ./clothes_dataset/brown_shoes/c224392e976d0e6711ba6cf6a247b5a16629b0ed.jpg  
      inflating: ./clothes_dataset/brown_shoes/c268dc6f92a712a254668177b15de4e1e564e68e.jpg  
      inflating: ./clothes_dataset/brown_shoes/c28455a3ddbb501e2636f2decd173f4954838639.jpg  
      inflating: ./clothes_dataset/brown_shoes/c431dace15f0e1eb5970dc61ee71518cdeaae93e.jpg  
      inflating: ./clothes_dataset/brown_shoes/c51fc82d14b652a5b02cd339b3e093a905e58daf.jpg  
      inflating: ./clothes_dataset/brown_shoes/c5945ee1b0a4225430b0e94050998fa0d7b7c260.jpg  
      inflating: ./clothes_dataset/brown_shoes/c5f678a4424373f3f82ef0ba5519c6cf8afda7b1.jpg  
      inflating: ./clothes_dataset/brown_shoes/c5f95be4b5335d7f9e0394348156cd065cfe20b6.jpg  
      inflating: ./clothes_dataset/brown_shoes/c61e37541f1c6bd4bb4b3eeb8093718019019093.jpg  
      inflating: ./clothes_dataset/brown_shoes/c6ff77670375dff0d9268146d96a86156e615260.jpg  
      inflating: ./clothes_dataset/brown_shoes/c71817b1ac37ac06b07cee38299f9fcd8d36b45d.jpg  
      inflating: ./clothes_dataset/brown_shoes/c792d963745162bc96bf7965c98ff69ef01f0728.jpg  
      inflating: ./clothes_dataset/brown_shoes/c858e22c0e77919eddbcee387c1f6109a0a9db91.jpg  
      inflating: ./clothes_dataset/brown_shoes/c8e302d8213efabf4c8752b28b9bacb4560bdc3c.jpg  
      inflating: ./clothes_dataset/brown_shoes/c9872ad4083036dbd33040ef09cce87b98a3e59f.jpg  
      inflating: ./clothes_dataset/brown_shoes/c9e51a1471739629e96a8ef6f56218164856c4d8.jpg  
      inflating: ./clothes_dataset/brown_shoes/ca4eed8f6f02b5afa77790610fee396d366fdc58.jpg  
      inflating: ./clothes_dataset/brown_shoes/ca71adfaeba329f5be0ed56332f5ebda0858215e.jpg  
      inflating: ./clothes_dataset/brown_shoes/cac9fa85658304798d78f2d57286e5b6599cddcb.jpg  
      inflating: ./clothes_dataset/brown_shoes/cacc4c5b74a2b44be7a6e91a9bd158b2f1e1bde5.jpg  
      inflating: ./clothes_dataset/brown_shoes/cb086f2ac25d4d067ec495902b42ed2ee41a3d1f.jpg  
      inflating: ./clothes_dataset/brown_shoes/cb4fbe02c9c3a5bf86fbbc6689f9d144f9381c86.jpg  
      inflating: ./clothes_dataset/brown_shoes/cd655f42e8cf693018e26f7697df9c42a098b4ba.jpg  
      inflating: ./clothes_dataset/brown_shoes/cd8e4a170a2e68e05060c75cd0be1e56d81cc402.jpg  
      inflating: ./clothes_dataset/brown_shoes/ce6ec74eb4dd1812ef1fe4877de24c12e2588372.jpg  
      inflating: ./clothes_dataset/brown_shoes/cf028adcf06b09d4b085fce034946b5633a68124.jpg  
      inflating: ./clothes_dataset/brown_shoes/cf1dd979438fe07cb89b4a9ef8519527529f9858.jpg  
      inflating: ./clothes_dataset/brown_shoes/cf412a54cc351dfc8e4ea0657a506ad73fb92c36.jpg  
      inflating: ./clothes_dataset/brown_shoes/cf4b656ecc992eead5b8aae1e9b3a60865125d6e.jpg  
      inflating: ./clothes_dataset/brown_shoes/cfa104f6f94e5186d4ea682b32bb626416e1b984.jpg  
      inflating: ./clothes_dataset/brown_shoes/cfd384746fa0c36e3731f5c00d3796c517222b6a.jpg  
      inflating: ./clothes_dataset/brown_shoes/d1175d94c52ae556f880fb3cbb57ccb20d1d284a.jpg  
      inflating: ./clothes_dataset/brown_shoes/d1bdb15c03d24acd4a61bb178bf9955d9c03c0cc.jpg  
      inflating: ./clothes_dataset/brown_shoes/d2733a9abe84bc20a19324f144d70948d1e4e17d.jpg  
      inflating: ./clothes_dataset/brown_shoes/d347a4953f2c6a107d9d2bf061ab5a551ece1555.jpg  
      inflating: ./clothes_dataset/brown_shoes/d34b448617e652a86590ad765fabb138578be31d.jpg  
      inflating: ./clothes_dataset/brown_shoes/d41b793b8736eaed72750877ff78ad2245d68e45.jpg  
      inflating: ./clothes_dataset/brown_shoes/d4ab69e71735a083937a71367c5fca1772a335a1.jpg  
      inflating: ./clothes_dataset/brown_shoes/d4dd546eb1def3c34bdf9b1fb79fbdea55f563fc.jpg  
      inflating: ./clothes_dataset/brown_shoes/d510fbf95a3fafdde0c487cf8cbf798c2001da5f.jpg  
      inflating: ./clothes_dataset/brown_shoes/d573b5269fc28d150366d7fbac918314d9597dfc.jpg  
      inflating: ./clothes_dataset/brown_shoes/d5954ea4e07e3ae43a8c477bc83f55e7cba93048.jpg  
      inflating: ./clothes_dataset/brown_shoes/d655e72e49069347bcba2b4ed472ea6a867caecc.jpg  
      inflating: ./clothes_dataset/brown_shoes/d6d443badcd2a686ff1dae6737f4d53c33c3f739.jpg  
      inflating: ./clothes_dataset/brown_shoes/d7d8ab68c06ea1ac245e6659714c4ea91a1acf2b.jpg  
      inflating: ./clothes_dataset/brown_shoes/d7d96a00e6d97b022806360bbee328e3eb340cbb.jpg  
      inflating: ./clothes_dataset/brown_shoes/d842b7a6a8fc7527a6d8fcd82ccc51c433041437.jpg  
      inflating: ./clothes_dataset/brown_shoes/d921b48090c2a5b626e510b875b453f665d626d6.jpg  
      inflating: ./clothes_dataset/brown_shoes/dab135299f654afdf3b77be74cb2a96579afd0c7.jpg  
      inflating: ./clothes_dataset/brown_shoes/db785bfa799728b37183cfdf2d2e43a74dd93d8c.jpg  
      inflating: ./clothes_dataset/brown_shoes/dc2be3475904d43bd0cb4790043e4ac9bca4b732.jpg  
      inflating: ./clothes_dataset/brown_shoes/dcc78131927ab0b30c292f0ff46d13c1cd92929f.jpg  
      inflating: ./clothes_dataset/brown_shoes/dd3f2560f6cf1713cb2a4c0c8286d620d85a1f16.jpg  
      inflating: ./clothes_dataset/brown_shoes/dd439806de09774dcf9a09eb02d80a5576e56a58.jpg  
      inflating: ./clothes_dataset/brown_shoes/dd940fe35d3531ed63fe4d4f3b8455d3d8ad7590.jpg  
      inflating: ./clothes_dataset/brown_shoes/ddeb6da2dad8f90960de3025b0389cea7135e23d.jpg  
      inflating: ./clothes_dataset/brown_shoes/de5a6ff31353f4e8025725b52dbbea82ab339d0b.jpg  
      inflating: ./clothes_dataset/brown_shoes/de96e011be3c074b004a53adb93756b6f948fbbe.jpg  
      inflating: ./clothes_dataset/brown_shoes/de9f1edc360e116d22cff7ef6f2f5c235f0f8121.jpg  
      inflating: ./clothes_dataset/brown_shoes/df1174e8a247fc179db138d675001d1819c6a2f1.jpg  
      inflating: ./clothes_dataset/brown_shoes/e24b81bdc6f3d03e4d17b5c3723d7afae83e89d9.jpg  
      inflating: ./clothes_dataset/brown_shoes/e25a85abe465b7e03c4b22969aae970a88dc1adf.jpg  
      inflating: ./clothes_dataset/brown_shoes/e2b88f05d43166d58a3dcb157fd91704155e356e.jpg  
      inflating: ./clothes_dataset/brown_shoes/e2cb41879d7b88f7968192182e0aa9ecf36ead46.jpg  
      inflating: ./clothes_dataset/brown_shoes/e2f8d9f18a1af7418874cec0147a3ae41d0254a6.jpg  
      inflating: ./clothes_dataset/brown_shoes/e35c7b3b9adabc408604669e0412fbd895a85f5c.jpg  
      inflating: ./clothes_dataset/brown_shoes/e54a0aa7896076f6ef4ff1de7db076c72c43c522.jpg  
      inflating: ./clothes_dataset/brown_shoes/e58599a32907432b2df4bba924e8fa67b526fe7e.jpg  
      inflating: ./clothes_dataset/brown_shoes/e5a08a2f0e64a69f7b2ac201febaf74d48408b54.jpg  
      inflating: ./clothes_dataset/brown_shoes/e680d29f60aaf31e227a146c238b2f351277f11b.jpg  
      inflating: ./clothes_dataset/brown_shoes/e6da55b9ebebc481e49380e10f34c69048b1493e.jpg  
      inflating: ./clothes_dataset/brown_shoes/e7086bf3822e87a16b22c8f6095d3f53c4577c3c.jpg  
      inflating: ./clothes_dataset/brown_shoes/e7f4564bbf84f07f1aacef00408c3f2aca678bf3.jpg  
      inflating: ./clothes_dataset/brown_shoes/e86ec1b7de9c378578bf7a463bef523ed2ff8234.jpg  
      inflating: ./clothes_dataset/brown_shoes/e8804ba90d2977738f50dacab8344a1a3748099d.jpg  
      inflating: ./clothes_dataset/brown_shoes/e8fc6bdba1dfa48b3d1302182bdeb7f9882be331.jpg  
      inflating: ./clothes_dataset/brown_shoes/e983c07bcb4a9ceb6e137b0e6af7c6d9f31e0a57.jpg  
      inflating: ./clothes_dataset/brown_shoes/e994f0ed41d8f5d4d817f39d47e6797a037afda8.jpg  
      inflating: ./clothes_dataset/brown_shoes/e99a8863f05304cacd3f7f98e9866392a142e78b.jpg  
      inflating: ./clothes_dataset/brown_shoes/e9e04aa98a52c2b1e73ff11445102347a6a47718.jpg  
      inflating: ./clothes_dataset/brown_shoes/e9e13a887326b618ac35f7ec4dab6d37732bd02a.jpg  
      inflating: ./clothes_dataset/brown_shoes/ea32507b06e6d0c43feea30f1e645d900dbde3b8.jpg  
      inflating: ./clothes_dataset/brown_shoes/eaa44d48def49a7b84826ff89ade996f85aadf12.jpg  
      inflating: ./clothes_dataset/brown_shoes/eb078e2e1ac6e60419a4bd4652365d5590ffa647.jpg  
      inflating: ./clothes_dataset/brown_shoes/ec324706a7b260c1d0566e6789425917fd11fb56.jpg  
      inflating: ./clothes_dataset/brown_shoes/ec8e2bdd774968e588a827d7b0ce588bfc0e853d.jpg  
      inflating: ./clothes_dataset/brown_shoes/ecab6b9ab6a1f527a6cd9914f99df8487714d406.jpg  
      inflating: ./clothes_dataset/brown_shoes/ed4149b703f434ad27a92be2c085cd3e30d6b737.jpg  
      inflating: ./clothes_dataset/brown_shoes/edaabd037b53937da042255e32f1ee601877183a.jpg  
      inflating: ./clothes_dataset/brown_shoes/eec3cc0d2befb82323cf33ef41a4ad2c9f05815d.jpg  
      inflating: ./clothes_dataset/brown_shoes/ef1805db79a48f86d171c9a5a434225095656563.jpg  
      inflating: ./clothes_dataset/brown_shoes/ef7b262f1c6229a3b0275c52a8d71eb193a15447.jpg  
      inflating: ./clothes_dataset/brown_shoes/ef93905f526e4cb72e65dfa46f143bf2dbdfe87a.jpg  
      inflating: ./clothes_dataset/brown_shoes/f01ccfb9dd2fe8bb4a3aea4dbbcafcb8f8cba0a1.jpg  
      inflating: ./clothes_dataset/brown_shoes/f11ac7c2c87b684051a64cf13f22175bec42bd09.jpg  
      inflating: ./clothes_dataset/brown_shoes/f1878ff63507a567a0ff10b1484d9c27c4620705.jpg  
      inflating: ./clothes_dataset/brown_shoes/f1a128fc0fdee779c053076f8bd1a51d56c45106.jpg  
      inflating: ./clothes_dataset/brown_shoes/f2000face4f7919aa7f2bafe6f2e3e8d82a66736.jpg  
      inflating: ./clothes_dataset/brown_shoes/f24eff27e3def1d2c83d5c3649678799ab3b9888.jpg  
      inflating: ./clothes_dataset/brown_shoes/f30dcdcaeefd34a6311f317093340240fbff7318.jpg  
      inflating: ./clothes_dataset/brown_shoes/f317ec322daa3fea476ef4768a6b18ff4e819e00.jpg  
      inflating: ./clothes_dataset/brown_shoes/f375b8171cac7ace332d09055629b0397c0619a3.jpg  
      inflating: ./clothes_dataset/brown_shoes/f3798c2c52d2655a3155aab9a5e041ac5645878f.jpg  
      inflating: ./clothes_dataset/brown_shoes/f3a10784cc5b4dab4bf7b23ca8dd5b9527fbbdaf.jpg  
      inflating: ./clothes_dataset/brown_shoes/f3c961ac26cfbde9e1459debe616b34b4390d4c5.jpg  
      inflating: ./clothes_dataset/brown_shoes/f4c28582ae170707946c70e7bf4734721cf3f15e.jpg  
      inflating: ./clothes_dataset/brown_shoes/f4ec19738405542763e9632059c7f13aec206b25.jpg  
      inflating: ./clothes_dataset/brown_shoes/f5adf7b19a35c52a793850ee4fafe92267735033.jpg  
      inflating: ./clothes_dataset/brown_shoes/f623fb4bd7c96a98fedf9dc67cf773ac3477e333.jpg  
      inflating: ./clothes_dataset/brown_shoes/f81ca8a9e0dca79348263a6a96072344e4ddc7fb.jpg  
      inflating: ./clothes_dataset/brown_shoes/f8374d4dafa0bb3fecaa97fa3d9ca549050773c2.jpg  
      inflating: ./clothes_dataset/brown_shoes/f87c3e6f53bccaa1341b5c654987f90ddf00d65a.jpg  
      inflating: ./clothes_dataset/brown_shoes/f91409c31269514d1c4e276d7b38d8adffd7f16e.jpg  
      inflating: ./clothes_dataset/brown_shoes/f99dcb2cfea4f03c9f16106349ad2504ecdfcb09.jpg  
      inflating: ./clothes_dataset/brown_shoes/fa0fc353e4ab060ef76cd2ee7374a8916b8e337c.jpg  
      inflating: ./clothes_dataset/brown_shoes/fb3eefdcaec98c75e93a6b859788f9e7bc8ba440.jpg  
      inflating: ./clothes_dataset/brown_shoes/fc042a986cb24638005660325762cc32276ef3ca.jpg  
      inflating: ./clothes_dataset/brown_shoes/fd28ce28044f206bfbecc094f72c389883d96fa2.jpg  
      inflating: ./clothes_dataset/brown_shoes/fe24082aa7037607dce771e87ff2f1352552b003.jpg  
      inflating: ./clothes_dataset/brown_shoes/ff228e6fd4c64b3a437023dfd7a84eab8e29ad07.jpg  
      inflating: ./clothes_dataset/brown_shoes/ffec0d5a59a770ba454686d7041a8a3ed11923da.jpg  
      inflating: ./clothes_dataset/brown_shorts/002a824a2d0026592e34e0df28ecd55abe71e57e.jpg  
      inflating: ./clothes_dataset/brown_shorts/00d1bd1878dcfffa2fb06eff0c429f4163dcd0a7.jpg  
      inflating: ./clothes_dataset/brown_shorts/0a1bddec088aff78a2babfd31db25d3ff1d3e0ef.jpg  
      inflating: ./clothes_dataset/brown_shorts/0e2b269e7c53d2c17526ec4da6c04cb6f7b4821d.jpg  
      inflating: ./clothes_dataset/brown_shorts/0e2fd256da17f80b048039a3ecb6305acdb8a0ba.jpg  
      inflating: ./clothes_dataset/brown_shorts/171ed97af2315bc067420f50f3ee9ac6672d7974.jpg  
      inflating: ./clothes_dataset/brown_shorts/2706208d880b7faba3b421d6575917208ff0d854.jpg  
      inflating: ./clothes_dataset/brown_shorts/34abaf486c9a11271f63587396c112664507c4d4.jpg  
      inflating: ./clothes_dataset/brown_shorts/34de2617006916a39951a9e6ae45b1ca0f9115f5.jpg  
      inflating: ./clothes_dataset/brown_shorts/38b88b3f22bca03f448bb7fde83848fd72757e4b.jpg  
      inflating: ./clothes_dataset/brown_shorts/3cd1c51eaca4fc3ace9a5be2498377988f1e53fa.jpg  
      inflating: ./clothes_dataset/brown_shorts/402d7e7aef043f066bcc756ea680ad678b62a4fe.jpg  
      inflating: ./clothes_dataset/brown_shorts/4e9229fa4eb94c5322a2a0677d02c87c02c31e7e.jpg  
      inflating: ./clothes_dataset/brown_shorts/4ee37372df01e2e2a94236c6ac2d2844ca66382f.jpg  
      inflating: ./clothes_dataset/brown_shorts/57f5c686d34cf0cfd46e5d7717e154c2892234fb.jpg  
      inflating: ./clothes_dataset/brown_shorts/5a646b38ff73cbde2b2c16263b4a24690cbd6131.jpg  
      inflating: ./clothes_dataset/brown_shorts/5ec4dabd3ea3901886eab27ea93801ec34fe77d9.jpg  
      inflating: ./clothes_dataset/brown_shorts/61d99410ec5e5c7bd26244e5bb7f798d5146e8ca.jpg  
      inflating: ./clothes_dataset/brown_shorts/6277913bf920305ac76a45e76c635d5c29453c3e.jpg  
      inflating: ./clothes_dataset/brown_shorts/7544664a962f9be48157e8ea88bae36ad94fdcea.jpg  
      inflating: ./clothes_dataset/brown_shorts/7d6f07610c51a1afdc576a858bd873c93948507c.jpg  
      inflating: ./clothes_dataset/brown_shorts/8e87278f0a71abdd9797f93cb5d710dd984e1224.jpg  
      inflating: ./clothes_dataset/brown_shorts/9a3c5e755f8aabae999f47c631bda8c7cf5d7e07.jpg  
      inflating: ./clothes_dataset/brown_shorts/9c562c24dd09765df1c9e5a98f25c3cc94ef9696.jpg  
      inflating: ./clothes_dataset/brown_shorts/a2442c72ef78ad54cdfb5107c5e61c5294828dd2.jpg  
      inflating: ./clothes_dataset/brown_shorts/a554a43bb31e5cff751c397c0f71ebe352c737d2.jpg  
      inflating: ./clothes_dataset/brown_shorts/a9e6182afd60497a496240b213ef4103e2b36a23.jpg  
      inflating: ./clothes_dataset/brown_shorts/b82cc06fde1d4f8d24d16d9515b2e3cf6553e936.jpg  
      inflating: ./clothes_dataset/brown_shorts/b9a013b20aa62bf28afc70df8b4dbbb5280b9403.jpg  
      inflating: ./clothes_dataset/brown_shorts/c4d75bbf1873ebc12e77770efc1d491299523696.jpg  
      inflating: ./clothes_dataset/brown_shorts/c676311f9b3a393875a0f802aac3c9e67cbde13b.jpg  
      inflating: ./clothes_dataset/brown_shorts/c8db9e0f7010592fa56f429a07142a360f95c9cd.jpg  
      inflating: ./clothes_dataset/brown_shorts/c9174d8bb94f49db586f520feb1391ca61895399.jpg  
      inflating: ./clothes_dataset/brown_shorts/ca0d8f76acf0daef2072b5d85c23a392e1708c94.jpg  
      inflating: ./clothes_dataset/brown_shorts/cac984f01729fb9c82b705519afe7a080bc47669.jpg  
      inflating: ./clothes_dataset/brown_shorts/d1d2daff64c64e1826df355272bead5f4b8c9085.jpg  
      inflating: ./clothes_dataset/brown_shorts/dd2c568194ea2dae35432805a49acbf24a7f634e.jpg  
      inflating: ./clothes_dataset/brown_shorts/e26f19884338f05fb0166367cc97591a098a90de.jpg  
      inflating: ./clothes_dataset/brown_shorts/efeaf60f92a7a8eb0282bd96bf05b954713cb20f.jpg  
      inflating: ./clothes_dataset/brown_shorts/f9f508136d8899d05ec671df0e7416ca6f661227.jpg  
      inflating: ./clothes_dataset/green_pants/002e7592988f007a267b7e259a154a2d1803db9f.jpg  
      inflating: ./clothes_dataset/green_pants/0156a7b0de03199881c9038d35a49e64bca92f86.jpg  
      inflating: ./clothes_dataset/green_pants/03088ac88d9f49f1fa145b3d5130e7ac01f3c686.jpg  
      inflating: ./clothes_dataset/green_pants/0343fa03880cb2783db000d1a74b80b9c510270b.jpg  
      inflating: ./clothes_dataset/green_pants/03d8e81b264cc294f80a3ccd9bd64db61198d331.jpg  
      inflating: ./clothes_dataset/green_pants/048fb3c98fac7811e45824d6f6311dc69dca52f3.jpg  
      inflating: ./clothes_dataset/green_pants/054c0703a79c9bda3bf50bb47c158cbaf37daa87.jpg  
      inflating: ./clothes_dataset/green_pants/05ad53e0be80cebbe7216725b37e144baf5f10fa.jpg  
      inflating: ./clothes_dataset/green_pants/074cfb8a95d57d0c9f4d4717446982e9282e9b5d.jpg  
      inflating: ./clothes_dataset/green_pants/079c3fe0248fa4861ae6f3dc61b7e0eed30478a7.jpg  
      inflating: ./clothes_dataset/green_pants/08ebe2836de3a90c13c665273ee80951cb831cb8.jpg  
      inflating: ./clothes_dataset/green_pants/0b2f9b6642f35e8939eacafd9c9c9ed2199732a6.jpg  
      inflating: ./clothes_dataset/green_pants/0ba1a31674aa11a4cd7da2a52941019ae8c10a43.jpg  
      inflating: ./clothes_dataset/green_pants/0bfaf2c00df035f1fbac2f69e238f3e1638e9cf4.jpg  
      inflating: ./clothes_dataset/green_pants/0e6df5d993e7744b77f140782e9c244e3748b341.jpg  
      inflating: ./clothes_dataset/green_pants/0ef17ad06d822f45c95645025bbf9821c6e85562.jpg  
      inflating: ./clothes_dataset/green_pants/0f89098738d781cb759eec983fc05f464d0cdd10.jpg  
      inflating: ./clothes_dataset/green_pants/10a1f6ae5b86fe9560868d69b30d6e3fe56e3fa1.jpg  
      inflating: ./clothes_dataset/green_pants/150d030ae4463c375582b12eae961659729bf3a7.jpg  
      inflating: ./clothes_dataset/green_pants/168d0f0c757876ba9aff0e09cfdf2fc150456eac.jpg  
      inflating: ./clothes_dataset/green_pants/16e5096530c29b9e1ce3e873e53cbd4be143e2b1.jpg  
      inflating: ./clothes_dataset/green_pants/18899a4c9e6cf31b971a39a136955f23dadb7697.jpg  
      inflating: ./clothes_dataset/green_pants/189d2ca1107828994e7d08090f381b990d5910f9.jpg  
      inflating: ./clothes_dataset/green_pants/1ab98b6b624dbfa4e1dedc01a8532cb245508d3b.jpg  
      inflating: ./clothes_dataset/green_pants/1b70c89a6c05382d6e6277db8acd405e7f0d7c4f.jpg  
      inflating: ./clothes_dataset/green_pants/1cc605878c7cfa72a2e0b7d859c13a70d369c439.jpg  
      inflating: ./clothes_dataset/green_pants/1d6a597f2bf7bd9a7c9f1cc598432b3e6bc6072d.jpg  
      inflating: ./clothes_dataset/green_pants/1f50020bb21160ab57e0539cbdd9591c7ad1913d.jpg  
      inflating: ./clothes_dataset/green_pants/209af8b9788d60f19a424e68f72aa11434978620.jpg  
      inflating: ./clothes_dataset/green_pants/244883f544b02bc84a82d6ef3de15873de7e598b.jpg  
      inflating: ./clothes_dataset/green_pants/25028dd48a159c25d29a9c6cac563a0703490858.jpg  
      inflating: ./clothes_dataset/green_pants/2673302df85748b1eff551ba9716e99aa2861737.jpg  
      inflating: ./clothes_dataset/green_pants/26ea98bcf98a70230ed9687ba350933a2039398a.jpg  
      inflating: ./clothes_dataset/green_pants/27eac4c1651c633589a9619370eef5ce55c128c8.jpg  
      inflating: ./clothes_dataset/green_pants/286ec9c7573b74096fde8c97551fad247485e37a.jpg  
      inflating: ./clothes_dataset/green_pants/29b0a53028413ba064ce13f03880a7d26e90c57d.jpg  
      inflating: ./clothes_dataset/green_pants/2a35ef0b5680e024f510aa53c784306e6f156ba2.jpg  
      inflating: ./clothes_dataset/green_pants/2a37dc349fa978327903964bb10040d01ebb2132.jpg  
      inflating: ./clothes_dataset/green_pants/2cedbc45ec89cd6ec1c76b14f40a09fa51284828.jpg  
      inflating: ./clothes_dataset/green_pants/2e1cb5def5561b582035d46f5004feb67563a431.jpg  
      inflating: ./clothes_dataset/green_pants/2f53a2d59b43af4580fa99fe0c3a66c5202e1b43.jpg  
      inflating: ./clothes_dataset/green_pants/2f96ca231902d5694356a86d9f9a9be2e51af03d.jpg  
      inflating: ./clothes_dataset/green_pants/308822af86a3c38b25ea64cc979b2d67d767c09e.jpg  
      inflating: ./clothes_dataset/green_pants/323464a2c2595626df21ac146e3911920bceab3c.jpg  
      inflating: ./clothes_dataset/green_pants/32e7e23645fcc84a857aa9c8394e7bb3a589155a.jpg  
      inflating: ./clothes_dataset/green_pants/32ee7e1cdd0d030be749393a01b6e1378e7fe7d4.jpg  
      inflating: ./clothes_dataset/green_pants/345050e04b7f11a69047f0962271648a97bd423c.jpg  
      inflating: ./clothes_dataset/green_pants/37d7da9d0b22e8b7e06bc73cbf67204583312d69.jpg  
      inflating: ./clothes_dataset/green_pants/38966f23c22ee98f3fd035d1eedd60b825c0f9ef.jpg  
      inflating: ./clothes_dataset/green_pants/38bc4fcb910aa246e67d86b9b58b2eda2c1a16a5.jpg  
      inflating: ./clothes_dataset/green_pants/38f5268c7c4b2fd5d010455ac574116516306f58.jpg  
      inflating: ./clothes_dataset/green_pants/3998182f32d9622c7d1f29def4b4277c8368171f.jpg  
      inflating: ./clothes_dataset/green_pants/3a2e79138abdabe733046809f872665eb88ab118.jpg  
      inflating: ./clothes_dataset/green_pants/3ae7658cd9c2206a15526e6bcb79ba633eee119f.jpg  
      inflating: ./clothes_dataset/green_pants/3bdddd9e91c09eaa39ec526178f4c57943352f42.jpg  
      inflating: ./clothes_dataset/green_pants/3d0d315ca3afd0e013ef9897ca6a5a4f7cde2c2e.jpg  
      inflating: ./clothes_dataset/green_pants/3f5e426b3720501d1f3af00bc73f9e23a02a7b04.jpg  
      inflating: ./clothes_dataset/green_pants/3feb43a0d6605fba235276a9f3276b9b39783d33.jpg  
      inflating: ./clothes_dataset/green_pants/40208771d2e2b0e939348b6d536ad6ba849485cf.jpg  
      inflating: ./clothes_dataset/green_pants/403ae42302ca9494777b026dc6fc5fd5923a276e.jpg  
      inflating: ./clothes_dataset/green_pants/414c6c03ac336dbb16904c340420f835c8d00c49.jpg  
      inflating: ./clothes_dataset/green_pants/4289cfbc4899973318cef572c7e5568f98de53f2.jpg  
      inflating: ./clothes_dataset/green_pants/456d8b491fedbafb6f15e977337982e600e2616c.jpg  
      inflating: ./clothes_dataset/green_pants/47d5bcadac9771362e2fca7c4ac858193354c5cd.jpg  
      inflating: ./clothes_dataset/green_pants/4810cb666110a5cab1492e365ef1a5f54b652356.jpg  
      inflating: ./clothes_dataset/green_pants/482b1f412d229e95ef446937d3f0f098399f417c.jpg  
      inflating: ./clothes_dataset/green_pants/49194d244218e23dd4e29afc9d12f8a04dd94f52.jpg  
      inflating: ./clothes_dataset/green_pants/4a1d06631e3108725fdc4425f0f1dc66755e6798.jpg  
      inflating: ./clothes_dataset/green_pants/4aa2d6045e83b351d62c35f6391118733c580b84.jpg  
      inflating: ./clothes_dataset/green_pants/4b3532e4749b59dbe44cba2d6f66eaa24265ea04.jpg  
      inflating: ./clothes_dataset/green_pants/4b7fa9f0e39a83ffe4c67c0b9ef9d3fd7b2c1415.jpg  
      inflating: ./clothes_dataset/green_pants/4ba6694fef68203a79dccde54555c8c20d77376a.jpg  
      inflating: ./clothes_dataset/green_pants/4c0066897eff1ab805a9cc48ab20581e83306b33.jpg  
      inflating: ./clothes_dataset/green_pants/4c03324180cae460f331bf32bb1e8937eaaa778b.jpg  
      inflating: ./clothes_dataset/green_pants/4c4dcebaebac6c74e09c210ee5782f63da370515.jpg  
      inflating: ./clothes_dataset/green_pants/4c81435c046b46486b7aac58a974bdfe70c37ef4.jpg  
      inflating: ./clothes_dataset/green_pants/4d5a31684f16ca75b2dfdd9284680d8206dfa3f7.jpg  
      inflating: ./clothes_dataset/green_pants/4d83d7ee6854b1577f4d5287cae6be0f23a33218.jpg  
      inflating: ./clothes_dataset/green_pants/4dd4076b79523014b88800fb80b49d9702e6a355.jpg  
      inflating: ./clothes_dataset/green_pants/4e8ce03d3a84d15c7536464af8d0bcb581a38c57.jpg  
      inflating: ./clothes_dataset/green_pants/4ed86df45829d8745a7cef5fb9d8f452cd6735f5.jpg  
      inflating: ./clothes_dataset/green_pants/4f7547659a2a2aa2f622c2112525b4ed70094c40.jpg  
      inflating: ./clothes_dataset/green_pants/51a67f9ef6ad24de1da10422defb5659df49d251.jpg  
      inflating: ./clothes_dataset/green_pants/51f1bf2ba10a258dc1b74c5d772a51ddca84f9fb.jpg  
      inflating: ./clothes_dataset/green_pants/52acaf5c03994bff6b68803577382b47f118a7e0.jpg  
      inflating: ./clothes_dataset/green_pants/52b88626283e336bc71c4694b41b45ec29510b1b.jpg  
      inflating: ./clothes_dataset/green_pants/52d8c01fcc55922127df4c71c8ee9c8a51e88533.jpg  
      inflating: ./clothes_dataset/green_pants/536177f04a1e62149d13ccd3cfb7beeed79c0eeb.jpg  
      inflating: ./clothes_dataset/green_pants/54bbff171e26d81a695ad090323d5fbd76a1a011.jpg  
      inflating: ./clothes_dataset/green_pants/55ff95943d587dffddfe87136b3e6656dd182e13.jpg  
      inflating: ./clothes_dataset/green_pants/5bd734cf22848d3cf91644562d01f84e2a4a08d7.jpg  
      inflating: ./clothes_dataset/green_pants/5c947d2ab7947d598c38ac54348dc0dc37b2c7a5.jpg  
      inflating: ./clothes_dataset/green_pants/5cd1980074ca6159d68c70fd0bb3ce9ee4674a32.jpg  
      inflating: ./clothes_dataset/green_pants/5d50b769277055d5b86255fa3789f2e05b00fc94.jpg  
      inflating: ./clothes_dataset/green_pants/5d8c149fbd71caf9bdfb235ec9188b21b842e82b.jpg  
      inflating: ./clothes_dataset/green_pants/5f9dfb9f73cbf370510463f449f32e9aa53033e0.jpg  
      inflating: ./clothes_dataset/green_pants/60df8ed27163215b65a77ff59bc3be1e74b0c38c.jpg  
      inflating: ./clothes_dataset/green_pants/612ebf8b922bf3d5ec52d66162e6ce47fb34881c.jpg  
      inflating: ./clothes_dataset/green_pants/62acb2036c817196491734f8f9f2a40cea875911.jpg  
      inflating: ./clothes_dataset/green_pants/6313df3c010a0ce05d1c4b7ee57c569a63d2502a.jpg  
      inflating: ./clothes_dataset/green_pants/645c875727220dbb7198ac0091fc1a9556a4b719.jpg  
      inflating: ./clothes_dataset/green_pants/66b75d9ac19bc009a87e8284538849a30b2dc7aa.jpg  
      inflating: ./clothes_dataset/green_pants/677cf008d4ff369aff2768b4945dc3ebf40d0d19.jpg  
      inflating: ./clothes_dataset/green_pants/68f44dcafe4cfaac68a3bd26825a6fd0382f5974.jpg  
      inflating: ./clothes_dataset/green_pants/6c6748250c34c803659d5cdb462642e8375db315.jpg  
      inflating: ./clothes_dataset/green_pants/6d9744c5bcf84888297b493d207db872b0aa7898.jpg  
      inflating: ./clothes_dataset/green_pants/7088ee62d1db2ed2636308041c08f76e23bb1748.jpg  
      inflating: ./clothes_dataset/green_pants/70f22cd2328455962e685d156550d3e13aa7038e.jpg  
      inflating: ./clothes_dataset/green_pants/716b38957183fd3571ef938c0f9ae68c88696ec8.jpg  
      inflating: ./clothes_dataset/green_pants/7316189c6ee05df650b20c95e3f0e85e42af2777.jpg  
      inflating: ./clothes_dataset/green_pants/73eea9d713183fffb0895d05ea2c119f732220ce.jpg  
      inflating: ./clothes_dataset/green_pants/744c4ae170537a29bda40d49413940901bd868e1.jpg  
      inflating: ./clothes_dataset/green_pants/76e8425fe1956cca5a9f94d841d17cd827e1d921.jpg  
      inflating: ./clothes_dataset/green_pants/7700b9c09b3f3db00985251b15adcbf92469f002.jpg  
      inflating: ./clothes_dataset/green_pants/77543a9b2fab7bc3c493e3188bc628e0f09c32d9.jpg  
      inflating: ./clothes_dataset/green_pants/776290aea580ffa6bb4e955f276877d13adabf29.jpg  
      inflating: ./clothes_dataset/green_pants/780f628efe1f9a6bec3ac51ba1e0a097227072a1.jpg  
      inflating: ./clothes_dataset/green_pants/787df39c63a039de49e6f069be88213d1840121c.jpg  
      inflating: ./clothes_dataset/green_pants/7b676d141c178675a732dbabe16f8a80982de2d0.jpg  
      inflating: ./clothes_dataset/green_pants/7c1f82c53c53abe1edbc132e87cc0297602ba97d.jpg  
      inflating: ./clothes_dataset/green_pants/7cdc740a39572d360652666f4cb94c244fe4d70a.jpg  
      inflating: ./clothes_dataset/green_pants/7efc0e549365d2eae52e0a525c597790a8da15f7.jpg  
      inflating: ./clothes_dataset/green_pants/7f40a5c9429a8605ca45ee9b09af9323095da3e1.jpg  
      inflating: ./clothes_dataset/green_pants/832e87aa239c3bbccafbdca2399db018e43271cb.jpg  
      inflating: ./clothes_dataset/green_pants/89053b8c26445d11975d4dccf7d48b03470e1fc0.jpg  
      inflating: ./clothes_dataset/green_pants/8a9a70198e96ca5af1111206b554722f57370483.jpg  
      inflating: ./clothes_dataset/green_pants/8b3355325a5514de112651998a775ae27558b1c5.jpg  
      inflating: ./clothes_dataset/green_pants/8b498ff3838edbdaa2ee90c88d9bccfd052e7088.jpg  
      inflating: ./clothes_dataset/green_pants/8b7e76e25769e8998dcc2be218f64bcaa0dc8138.jpg  
      inflating: ./clothes_dataset/green_pants/8f8b5abfd79b30849438bdbd15b11225b4977f58.jpg  
      inflating: ./clothes_dataset/green_pants/9252a5ccd8159d57d45f3c3fed0ebe0e819c7c8f.jpg  
      inflating: ./clothes_dataset/green_pants/95c57ca084279afe34656b2ad2566c5607192974.jpg  
      inflating: ./clothes_dataset/green_pants/962e25375b677b310a92fe16ad84d1739e0446d6.jpg  
      inflating: ./clothes_dataset/green_pants/964688b68469381ef21dc5869a1a3c64607e983f.jpg  
      inflating: ./clothes_dataset/green_pants/968752be186663eb463064e5f23ef8c692a19227.jpg  
      inflating: ./clothes_dataset/green_pants/96d55a6021c57d21092c917af1a898534318c663.jpg  
      inflating: ./clothes_dataset/green_pants/98c6aa1c43712e6c3ee7997166d785a835412e44.jpg  
      inflating: ./clothes_dataset/green_pants/99eeb0d1e429924cba4cc967ae9afb386d1d65b4.jpg  
      inflating: ./clothes_dataset/green_pants/9afe582b3dffe3d20428fb1711d2877a24dc00b1.jpg  
      inflating: ./clothes_dataset/green_pants/9b7d74e89eaf8381e123f560027b33c1f1c41d0a.jpg  
      inflating: ./clothes_dataset/green_pants/9c6573dd2115a1c38465fc45904c5dbb05956bff.jpg  
      inflating: ./clothes_dataset/green_pants/9c9cad99b00400e4a14c68944e67c2dd450a95cc.jpg  
      inflating: ./clothes_dataset/green_pants/9e364ac1ecf16bd7c263fb0c006131aed1e1fc50.jpg  
      inflating: ./clothes_dataset/green_pants/9e5dc7c6520689cfcecb71c846323a9a14ecccda.jpg  
      inflating: ./clothes_dataset/green_pants/9f735d4675884932a031877a5d2bf6d5013ec857.jpg  
      inflating: ./clothes_dataset/green_pants/a11f74593c2e36a9960f18d75f97f39ffcbec7af.jpg  
      inflating: ./clothes_dataset/green_pants/a2425eb1ac10aa6072bb5e004f3e10931c4f5923.jpg  
      inflating: ./clothes_dataset/green_pants/a36aecfe24782a1224248ffe716302e4d22c195b.jpg  
      inflating: ./clothes_dataset/green_pants/a4d01c0121a246e3fff50a34952c5a44a67e15d5.jpg  
      inflating: ./clothes_dataset/green_pants/a68ddd8a6f45ba18486e793ff2e4c90c62cc2d81.jpg  
      inflating: ./clothes_dataset/green_pants/a7cd6fb28a0ebf81458a299fe19cd0e0764e512a.jpg  
      inflating: ./clothes_dataset/green_pants/a823894164813057b323333d1e0a4653c5c54cf6.jpg  
      inflating: ./clothes_dataset/green_pants/a8da02948681dd5701890120d7dc29adcd900a2e.jpg  
      inflating: ./clothes_dataset/green_pants/a9cce09538692373b3822953afd4925aa9a9c540.jpg  
      inflating: ./clothes_dataset/green_pants/a9fb88d2f035b59f90cd44218b5e327bb2ef340d.jpg  
      inflating: ./clothes_dataset/green_pants/ab234a4a0e0ba3e621f5b2748916aeaec415afd5.jpg  
      inflating: ./clothes_dataset/green_pants/af91f148fd9d806f8a9aa468599d0c442caea642.jpg  
      inflating: ./clothes_dataset/green_pants/b12ff136ba0397ab72759e83093a743eaa633eb3.jpg  
      inflating: ./clothes_dataset/green_pants/b2378963e58b6e5960eb5f5350c6b4845a236f6a.jpg  
      inflating: ./clothes_dataset/green_pants/b34dccdfac551f5f585e94471529d6f0dfbf753e.jpg  
      inflating: ./clothes_dataset/green_pants/b41c4e5c03f843fa8fb3a8a3f2d4fd29de38ac89.jpg  
      inflating: ./clothes_dataset/green_pants/b650016b00c5aecda97a523c6dd4e6b67e6c833c.jpg  
      inflating: ./clothes_dataset/green_pants/b67a905dbe74f912ba7f30f724f16a5819ba4f66.jpg  
      inflating: ./clothes_dataset/green_pants/b7a5de5f448e02b9d830eafc0c80e2a0bf38f67d.jpg  
      inflating: ./clothes_dataset/green_pants/b853669d13fcc1ba310ff18a2e82dd0808797964.jpg  
      inflating: ./clothes_dataset/green_pants/bba6f680d6e45bc6c3df27a9e87c7f4f7c536c3a.jpg  
      inflating: ./clothes_dataset/green_pants/bd06b640524d74abe9bf7d8528b4b7d174cf7009.jpg  
      inflating: ./clothes_dataset/green_pants/be9886a7fd17693d0ae31a706d2fe24a4ccce61c.jpg  
      inflating: ./clothes_dataset/green_pants/bf319accf0029b3f306aa77b9763cbbbd17dd7bd.jpg  
      inflating: ./clothes_dataset/green_pants/c05c106642395c1a3147d47863ec70914b0d5bdd.jpg  
      inflating: ./clothes_dataset/green_pants/c39dd1326574d582a2279dd5151d8bfb76e93e76.jpg  
      inflating: ./clothes_dataset/green_pants/c505838dddb8f87db0dfcc7e2f8b227b9f37a035.jpg  
      inflating: ./clothes_dataset/green_pants/c6d8699e2f06aaba3e43726c16545d21a98eb947.jpg  
      inflating: ./clothes_dataset/green_pants/c6ffe2923b1ae784c00d1b5be73776c6dafad0ab.jpg  
      inflating: ./clothes_dataset/green_pants/c7d3bcbc8958cacd7e04efba6384ab899a8bec94.jpg  
      inflating: ./clothes_dataset/green_pants/c7ef382b413ac00339c60a5876b64d20d0796dd6.jpg  
      inflating: ./clothes_dataset/green_pants/cb9ad59cfc6e9fa48dfbeecfec6ce7a4040a755e.jpg  
      inflating: ./clothes_dataset/green_pants/cc372fdbf7ea42a321276552b812818471ebdcae.jpg  
      inflating: ./clothes_dataset/green_pants/cfe8798c27c1a3470ab51d99e9b9d9e51d2fe758.jpg  
      inflating: ./clothes_dataset/green_pants/d0730a915f3bd53999c697d4c358f1508d950d78.jpg  
      inflating: ./clothes_dataset/green_pants/d1ec15f24cf8c61e302c99e3198efa6f762e003b.jpg  
      inflating: ./clothes_dataset/green_pants/d1f90b65324a646b4185d58b67cd8420cc623d48.jpg  
      inflating: ./clothes_dataset/green_pants/d22bda2f8448cb8e16edad6401e7ddd550b0d5e2.jpg  
      inflating: ./clothes_dataset/green_pants/d3038144e20146257c7fa0aa127d0b15cef5a05e.jpg  
      inflating: ./clothes_dataset/green_pants/d35fbce781121c0149d2222491b635fbae0474c3.jpg  
      inflating: ./clothes_dataset/green_pants/d7179545387718187be41383e6a8f358455aef19.jpg  
      inflating: ./clothes_dataset/green_pants/d790feb5d4b3939ecccbfbe3b2f7e75df714283a.jpg  
      inflating: ./clothes_dataset/green_pants/d8e486a4f0f6bacc9a662677a6159e18fceb186a.jpg  
      inflating: ./clothes_dataset/green_pants/d98d6858fb2293316e6ce20604cb7a1342034b9b.jpg  
      inflating: ./clothes_dataset/green_pants/daf88d238c948fd22ccb94ac6df90954feccfd77.jpg  
      inflating: ./clothes_dataset/green_pants/db5959781c15f775335ec1d1cbf0e3d334139f18.jpg  
      inflating: ./clothes_dataset/green_pants/db5dc885c1c7ebc1aff6e7773308f169c5dbce2f.jpg  
      inflating: ./clothes_dataset/green_pants/dc1da052f3d0c10f8416f241c86de702524c7274.jpg  
      inflating: ./clothes_dataset/green_pants/dd0b72fba812d0debc48755469c365ef74874b25.jpg  
      inflating: ./clothes_dataset/green_pants/dd36abc61daf93d7b073b66c473bcb4d52b06320.jpg  
      inflating: ./clothes_dataset/green_pants/de556b540f6390147bb80ae3ce958ac5677c8145.jpg  
      inflating: ./clothes_dataset/green_pants/dff94b7c5693032d72c70064009bc51cc2434167.jpg  
      inflating: ./clothes_dataset/green_pants/e0c9507fef0da92c8b1f5b6f8f5310115581d994.jpg  
      inflating: ./clothes_dataset/green_pants/e22b1c6a2cf0fc4fb2dc87ec8bd43db9fab7ca76.jpg  
      inflating: ./clothes_dataset/green_pants/e44b4a51cb52a9cbb52579cf222487375d5cc2fb.jpg  
      inflating: ./clothes_dataset/green_pants/e46aa2ac11b751f7e77ef276caf45f22b77646e4.jpg  
      inflating: ./clothes_dataset/green_pants/e4ea4f62be803249b6d0251a2ce68189d0748a59.jpg  
      inflating: ./clothes_dataset/green_pants/e5c6b947b6e153727b83836ed1fd8dc36143f82f.jpg  
      inflating: ./clothes_dataset/green_pants/e73cac7ac5a67b1be4f648bfeb05dceba4ca0135.jpg  
      inflating: ./clothes_dataset/green_pants/eb023619be3423383e3a22c1ef50e553dd286c14.jpg  
      inflating: ./clothes_dataset/green_pants/eb51fdd99aa31c298a792da46538b73e186b1324.jpg  
      inflating: ./clothes_dataset/green_pants/ebd63541ead70eff83d7ac57546533f531c13d5a.jpg  
      inflating: ./clothes_dataset/green_pants/ebf757a60f22d892f090d8d612623a9f60cadb9d.jpg  
      inflating: ./clothes_dataset/green_pants/ec543ca241cefb2b3a52fb6b0b27ed3164d22a47.jpg  
      inflating: ./clothes_dataset/green_pants/eff74dd105075001c738f126689447ff10c1efce.jpg  
      inflating: ./clothes_dataset/green_pants/f05f90a61b962e8a08fd676cf6df2c6e1abb3429.jpg  
      inflating: ./clothes_dataset/green_pants/f17b9075df61675e7b10903222f4db4c6d50b28d.jpg  
      inflating: ./clothes_dataset/green_pants/f216d6d4a2d1efe5d307fa9f442f06d48fe0b6cf.jpg  
      inflating: ./clothes_dataset/green_pants/f2abe79537c3fe40e0d0c14964778eb569d85cab.jpg  
      inflating: ./clothes_dataset/green_pants/f2dbeeeea637de33ca2dc5925fb70d94763c6b39.jpg  
      inflating: ./clothes_dataset/green_pants/f3c587828f5622b8c63f473be4fa914bf28559de.jpg  
      inflating: ./clothes_dataset/green_pants/f4aa6140718a3bc2b1c393ac92f67c6c14de50ec.jpg  
      inflating: ./clothes_dataset/green_pants/f50f46d5a0a8815a17b56d65db7b146e2ec9c6c9.jpg  
      inflating: ./clothes_dataset/green_pants/f6c6d0bb37dda0c9602bfa482487e0c49648eb30.jpg  
      inflating: ./clothes_dataset/green_pants/f9120a5a44078b2111f9d36e5cca31f2f8a2344d.jpg  
      inflating: ./clothes_dataset/green_pants/f9e4c6da1a78732f6b3bf7ab4f65845c42f979d9.jpg  
      inflating: ./clothes_dataset/green_pants/fb74a6726c8898c42f4bef3b0a8b26cf377756a1.jpg  
      inflating: ./clothes_dataset/green_pants/fb868d8a8a58b164a99af3d75bffe49b60237533.jpg  
      inflating: ./clothes_dataset/green_pants/fd4a60e50d2057e49e2f1b3277dc960721da06ad.jpg  
      inflating: ./clothes_dataset/green_pants/fd913ed789db648116356d9a3c2523d3642df34a.jpg  
      inflating: ./clothes_dataset/green_pants/feedf18f8782613dc33a2f94e0b99d16be080d3c.jpg  
      inflating: ./clothes_dataset/green_pants/ff8232646270ef6b4109daa0db452ed81899b644.jpg  
      inflating: ./clothes_dataset/green_shirt/002345bfdd288450d742aa31b72d1a8f13bd07de.jpg  
      inflating: ./clothes_dataset/green_shirt/01016cfb0d2eeea529ceeb6a12d17b8dbd23823d.jpg  
      inflating: ./clothes_dataset/green_shirt/0271d0189e6487b2f60978bdc0a769d259df8a25.jpg  
      inflating: ./clothes_dataset/green_shirt/02d556f08945e898222f172ef91b00f985c5d389.jpg  
      inflating: ./clothes_dataset/green_shirt/03bf78e45867901eb73ce149d67f1e29860d28c4.jpg  
      inflating: ./clothes_dataset/green_shirt/03fb3285b8a5b5f2a1acc8116f87917a557dab00.jpg  
      inflating: ./clothes_dataset/green_shirt/04de17cadd9fddd2f6454174dc764a8e6b03665f.jpg  
      inflating: ./clothes_dataset/green_shirt/05c6dfb47f585cfb9b926e65beb4f52645690325.jpg  
      inflating: ./clothes_dataset/green_shirt/069dcd9d80c0b7bb71acba6c04865b01dd325ecc.jpg  
      inflating: ./clothes_dataset/green_shirt/087bb703ed504a52ca909e59219f6815b75447c9.jpg  
      inflating: ./clothes_dataset/green_shirt/0ba7006f64a7c8fd11cfd1b79e68efb71b77359d.jpg  
      inflating: ./clothes_dataset/green_shirt/0c2dd098f8705a183e07513699e6f3801d14e00f.jpg  
      inflating: ./clothes_dataset/green_shirt/0cc2bff732c25853a3ebc9ce9c4ecdd270e7f61b.jpg  
      inflating: ./clothes_dataset/green_shirt/0d9600aa105cd2bfbc39dd6b2f26ec853148ac64.jpg  
      inflating: ./clothes_dataset/green_shirt/0e036d8f302d6e2cbc05779122969a9e32d8d183.jpg  
      inflating: ./clothes_dataset/green_shirt/0f17176d59ec5f2a8963860b25f23413b49aa6f6.jpg  
      inflating: ./clothes_dataset/green_shirt/0f99508584911a86d5a03a2ee830c9cfb278a95b.jpg  
      inflating: ./clothes_dataset/green_shirt/0ff8b2a2fa2b112b0336677c3cff2f0f2a89a72d.jpg  
      inflating: ./clothes_dataset/green_shirt/1080a927239fe232d5c9a0a2cd811ea013475f7b.jpg  
      inflating: ./clothes_dataset/green_shirt/108f737d1d2d349343125308a0da3666db4f9785.jpg  
      inflating: ./clothes_dataset/green_shirt/11fb6df4927a327af064ce5403442d4239fb0536.jpg  
      inflating: ./clothes_dataset/green_shirt/126986cb2da49848231b88a2a136d2c25a710b78.jpg  
      inflating: ./clothes_dataset/green_shirt/1465c71244bc876b17c89dbd1f85712407e35b40.jpg  
      inflating: ./clothes_dataset/green_shirt/16c061b04fc4b3f39281798ff854576777744182.jpg  
      inflating: ./clothes_dataset/green_shirt/170d47884dd250d167617984c3a457b87e76e1ed.jpg  
      inflating: ./clothes_dataset/green_shirt/172b0f82f1ecd4d83a8c02756074b211f5f00c27.jpg  
      inflating: ./clothes_dataset/green_shirt/17a8e1aa97a72687ae4ff4027b27bf8273e870f5.jpg  
      inflating: ./clothes_dataset/green_shirt/1954cc1d63b87aa2d459dca4075812cf106f915c.jpg  
      inflating: ./clothes_dataset/green_shirt/19ed1dd7b4cf01b301b2f6e970eb90edfdbb7951.jpg  
      inflating: ./clothes_dataset/green_shirt/1a26a7213218bc434a9f68b9cb9cecc1ef4303ce.jpg  
      inflating: ./clothes_dataset/green_shirt/1a96ea968cb29a18a67e3b3152e20ecc44f790cd.jpg  
      inflating: ./clothes_dataset/green_shirt/1ba50f3abe8f9155190522d9daa085e965d4e0b0.jpg  
      inflating: ./clothes_dataset/green_shirt/1bf4fe44e850ae6b5a9036a099030facdbbe49cb.jpg  
      inflating: ./clothes_dataset/green_shirt/1c009886f7e22708d350e89c9c9d254da7b8ec24.jpg  
      inflating: ./clothes_dataset/green_shirt/1dc5bb201a5944640f165b4e0c24bafa14c7d43c.jpg  
      inflating: ./clothes_dataset/green_shirt/1e0c4fe2254097079fcb23b2f29d9f7ad992209f.jpg  
      inflating: ./clothes_dataset/green_shirt/1e105e0c344a59fc7f9b2c8eddf1476e33f5587d.jpg  
      inflating: ./clothes_dataset/green_shirt/1ec47710917621a26bc4a67268acac34d5cfce5d.jpg  
      inflating: ./clothes_dataset/green_shirt/20616a5d2503328bd15ff2052a66f19ce454bbc9.jpg  
      inflating: ./clothes_dataset/green_shirt/226fb6caf599cb1041d2107ca4bd5c79cfa32223.jpg  
      inflating: ./clothes_dataset/green_shirt/241d8eaac017d7ff35dfae2b1ee772675cc92a85.jpg  
      inflating: ./clothes_dataset/green_shirt/24269ccfc2886220b4c13052099ddb4ec0b1d0b9.jpg  
      inflating: ./clothes_dataset/green_shirt/24421ae95818638886c427c9dd0a7223196f5822.jpg  
      inflating: ./clothes_dataset/green_shirt/2447299599dd64c275bdeebbb9b1daac58b3951c.jpg  
      inflating: ./clothes_dataset/green_shirt/25be16314b50726277c096c917469c61a813935a.jpg  
      inflating: ./clothes_dataset/green_shirt/25e27ffbe79e4a6c2ad2c22e7794de461b28f9d5.jpg  
      inflating: ./clothes_dataset/green_shirt/26f8c009c126e8cecd6b65ce95b789d170a92a6c.jpg  
      inflating: ./clothes_dataset/green_shirt/27462a20c64b506339d74a1b596c80291bf9f2d7.jpg  
      inflating: ./clothes_dataset/green_shirt/284b658dd23c2c6cc898c4e27049475881e48b21.jpg  
      inflating: ./clothes_dataset/green_shirt/2889a0bb7be84731782468c814df6eadcad149e5.jpg  
      inflating: ./clothes_dataset/green_shirt/2af6dab34ee2df82d2a6f9eb0910a47b453bf429.jpg  
      inflating: ./clothes_dataset/green_shirt/2e6d0637fdefcb249807e187095358363f461b8f.jpg  
      inflating: ./clothes_dataset/green_shirt/2f7e16a11b49f8e3ecc1defef0c16a5ff5ff740b.jpg  
      inflating: ./clothes_dataset/green_shirt/2f8e38ce873daceb55063d0dd86350021026c1a6.jpg  
      inflating: ./clothes_dataset/green_shirt/2ff84d90f8f51596167958bfa918fb83820b5feb.jpg  
      inflating: ./clothes_dataset/green_shirt/3102ac904528347dd7112f47a3ed096671c07d19.jpg  
      inflating: ./clothes_dataset/green_shirt/32514a0a15c595c623cf967b3e1e111362646f17.jpg  
      inflating: ./clothes_dataset/green_shirt/32b28342469738e8e4c9919e1c44d83fc5228571.jpg  
      inflating: ./clothes_dataset/green_shirt/3409fa5914c26d1b4f4219c4e57f1a75eee17d3e.jpg  
      inflating: ./clothes_dataset/green_shirt/342c6f3d0fcd5039eaa62b2de5845ea3d8429d95.jpg  
      inflating: ./clothes_dataset/green_shirt/34aefdc161f76555617d1be7ef6840421112e78c.jpg  
      inflating: ./clothes_dataset/green_shirt/36e1881264afe4b54a17066b2a46a5dcf0daf6c4.jpg  
      inflating: ./clothes_dataset/green_shirt/3856e8d852296f615aa93f118deb7045f314280a.jpg  
      inflating: ./clothes_dataset/green_shirt/38a3f94ba523f117f65761ccca052099e5dda3c0.jpg  
      inflating: ./clothes_dataset/green_shirt/39af05393b363fe1773528b2f83e8f47049ec5f6.jpg  
      inflating: ./clothes_dataset/green_shirt/3b296c0aa9f3489c13a882f68223e7d15801a526.jpg  
      inflating: ./clothes_dataset/green_shirt/3b3a82117a3419d04f59246995d048ee31b2bcf7.jpg  
      inflating: ./clothes_dataset/green_shirt/3b55b8c2577def4a2f8013ec15c49c8400b93b44.jpg  
      inflating: ./clothes_dataset/green_shirt/3b6d735f4d412e8f3ff8a3361e3aff58ff8be3ee.jpg  
      inflating: ./clothes_dataset/green_shirt/3c5876b6e55d637bc92f9cb86724c5c53560135d.jpg  
      inflating: ./clothes_dataset/green_shirt/3f555570173e2aa76a27b72d3a0c0cb0b8502959.jpg  
      inflating: ./clothes_dataset/green_shirt/3ff21c1ee21c230c6c73a02c455caeceb9b85b94.jpg  
      inflating: ./clothes_dataset/green_shirt/4324ab01cb73e26a0febab9ee1217c8d201da0d4.jpg  
      inflating: ./clothes_dataset/green_shirt/448597c3e32123ab100241b1d69f03b97f877020.jpg  
      inflating: ./clothes_dataset/green_shirt/44c7b90cef00ed533bb58bef7e49a78ee5511fa5.jpg  
      inflating: ./clothes_dataset/green_shirt/45216497e1b7192c8afba9c7e0055555021b5580.jpg  
      inflating: ./clothes_dataset/green_shirt/453c090fb4504ad7d6a9cc700458f7a617ceab77.jpg  
      inflating: ./clothes_dataset/green_shirt/47dfa624e162e5446af01f04da7f84df48b2659f.jpg  
      inflating: ./clothes_dataset/green_shirt/4829da4df1c23969f79a715daff71ab056454694.jpg  
      inflating: ./clothes_dataset/green_shirt/48eec1be592c87c51d0f78dac6626cad3fb3ff09.jpg  
      inflating: ./clothes_dataset/green_shirt/495d349f8f446742543ba3955fbe03aa91f90885.jpg  
      inflating: ./clothes_dataset/green_shirt/4b79d5266b89d0ac8fa4319d972289a9326514a3.jpg  
      inflating: ./clothes_dataset/green_shirt/4c8fdff77de26b82190fcfde7d07d1881a5d1974.jpg  
      inflating: ./clothes_dataset/green_shirt/4cea08508546bb33cc886ad7f6b06f6085bb66a1.jpg  
      inflating: ./clothes_dataset/green_shirt/4d07f52009ff4fba2906e3df64992f9281f3369b.jpg  
      inflating: ./clothes_dataset/green_shirt/4d22ce842a267ce19f6198aab2b8c5629cb81685.jpg  
      inflating: ./clothes_dataset/green_shirt/4d5ddc9d586e0683b7e19373ecfd0ff455e2f336.jpg  
      inflating: ./clothes_dataset/green_shirt/52a4fec8e4a8dabb48558590499ad31696c2b43e.jpg  
      inflating: ./clothes_dataset/green_shirt/52eb05ee3540376c934569280062701a34d8cebc.jpg  
      inflating: ./clothes_dataset/green_shirt/531c297f0678fcaca3db1a8364404099c5691f56.jpg  
      inflating: ./clothes_dataset/green_shirt/533e319d35371761bd3952d2f283b6022ca44939.jpg  
      inflating: ./clothes_dataset/green_shirt/570b2b1138ef62058290b9934b22d8d19066b82c.jpg  
      inflating: ./clothes_dataset/green_shirt/5786316341c2bcb91d2763ec52cb169dc12c0ae4.jpg  
      inflating: ./clothes_dataset/green_shirt/57f459de730c1e5585982df9a9a0216fde721479.jpg  
      inflating: ./clothes_dataset/green_shirt/595ec3e6866f0e0ce15bf775c1d864254cc837b9.jpg  
      inflating: ./clothes_dataset/green_shirt/5a1247b420f67634602f86d586c6ec7c4fa47da5.jpg  
      inflating: ./clothes_dataset/green_shirt/5d1243c52eb66ebf9ed968f136a02f6f36764613.jpg  
      inflating: ./clothes_dataset/green_shirt/5ed96aa9601b6fac09b9fe8ec0cfe5a9bf52e99a.jpg  
      inflating: ./clothes_dataset/green_shirt/5f48ed8f307b175768502ea55fe0f60f53fdae0f.jpg  
      inflating: ./clothes_dataset/green_shirt/60b0ea02d7068f0ad38e65732ee4f868be23dc54.jpg  
      inflating: ./clothes_dataset/green_shirt/65051d8f1a6ad455ecf9c815b6ee62c7b9eaaf72.jpg  
      inflating: ./clothes_dataset/green_shirt/67420a88d7bf385e4b5fadd5af10a4598cc54e58.jpg  
      inflating: ./clothes_dataset/green_shirt/6786448ebf255639106fae7fc1d1672d67893aa7.jpg  
      inflating: ./clothes_dataset/green_shirt/68eb0e69ae54143b922a46f614037835b5f14820.jpg  
      inflating: ./clothes_dataset/green_shirt/69a94d3976e49d151fa3d24aaa31d8e253c5832e.jpg  
      inflating: ./clothes_dataset/green_shirt/6ad1ce04d88f861f6fc13642dddc22ae5d0ec15c.jpg  
      inflating: ./clothes_dataset/green_shirt/6b8fd7c8bcb141317303e0495b8d3b62cc873bd2.jpg  
      inflating: ./clothes_dataset/green_shirt/6b92b738cf192771e650fa1dfbe6fddabbdf30d4.jpg  
      inflating: ./clothes_dataset/green_shirt/6bac923e42f3a600686e1c75ad08555d17fc0640.jpg  
      inflating: ./clothes_dataset/green_shirt/6bb63a88ba6606735bc505bfe084805d4a2bab99.jpg  
      inflating: ./clothes_dataset/green_shirt/6bc998de71963ef364ee6778e209cd8c7117a590.jpg  
      inflating: ./clothes_dataset/green_shirt/6e71f01b3eb30de1effd2b1b90ea564e7fdff2ee.jpg  
      inflating: ./clothes_dataset/green_shirt/6f5dcd21411ea6e5f50b880324422b58a02ca2f6.jpg  
      inflating: ./clothes_dataset/green_shirt/7066aee24ba61d20cda91e00821652999c56e065.jpg  
      inflating: ./clothes_dataset/green_shirt/71421c7dd5430a49d29fb8749710933d23bec12b.jpg  
      inflating: ./clothes_dataset/green_shirt/73cd3fb09d658d4422f171d51f9946e3ce934466.jpg  
      inflating: ./clothes_dataset/green_shirt/7447725d443413e28fb65b97a671c177ff963514.jpg  
      inflating: ./clothes_dataset/green_shirt/74817dc816f8295aeaf81599a96df2dc58a81011.jpg  
      inflating: ./clothes_dataset/green_shirt/750dfe7f11ac9cdcb64c061c7adf582746597d29.jpg  
      inflating: ./clothes_dataset/green_shirt/776355209f4068a536b44ce3405526c38ebce912.jpg  
      inflating: ./clothes_dataset/green_shirt/77d0801cdba0966f10d6551dbb21c56376852416.jpg  
      inflating: ./clothes_dataset/green_shirt/783273863668afca6e7d7b91d17a781caa5f0cc1.jpg  
      inflating: ./clothes_dataset/green_shirt/7bc923300eaf320b7a60b48ab3ae5cfcdee2d053.jpg  
      inflating: ./clothes_dataset/green_shirt/7be93c0fe39fd61ea0c5217ea9870d5163d33098.jpg  
      inflating: ./clothes_dataset/green_shirt/7c55d310b89d2361ab5596407f9a5ab5db556c97.jpg  
      inflating: ./clothes_dataset/green_shirt/7d6725d497f2acb4118b6e0d62929ea3fc18a5a1.jpg  
      inflating: ./clothes_dataset/green_shirt/7eea97fc18a34e94743173ef226411e61671310d.jpg  
      inflating: ./clothes_dataset/green_shirt/7eed7efa14d39dc9f682d1250554aec20ebebc7c.jpg  
      inflating: ./clothes_dataset/green_shirt/81bb6ecdd1f62912c28ba8252d10e4ac157cb1bd.jpg  
      inflating: ./clothes_dataset/green_shirt/85672b15a863afc483c9add033e4170351d1b987.jpg  
      inflating: ./clothes_dataset/green_shirt/86131e6a08efde502b4a55c0b5f5fb1f7ecabce7.jpg  
      inflating: ./clothes_dataset/green_shirt/86c4d770cd8f0c94c9e8fe318485d028574f5d66.jpg  
      inflating: ./clothes_dataset/green_shirt/87ca6631ed8545934e3a62181a8bab4859f9fda7.jpg  
      inflating: ./clothes_dataset/green_shirt/87eb3773737e874eed8cf561d07b8bf8d5637ad7.jpg  
      inflating: ./clothes_dataset/green_shirt/899c0086bfd606be04851701f98d6de56603cf25.jpg  
      inflating: ./clothes_dataset/green_shirt/8b9b9ca5fd3296e79bb57414a39eab9fb3f86750.jpg  
      inflating: ./clothes_dataset/green_shirt/8fc62b0da30117cc5da66192d521d02c7e2b6da8.jpg  
      inflating: ./clothes_dataset/green_shirt/90c5db4943cca7a28de355a68246199d79b859b7.jpg  
      inflating: ./clothes_dataset/green_shirt/915c509883a9b65d343fb7411c973f11d953bd04.jpg  
      inflating: ./clothes_dataset/green_shirt/91924aef347adbfea92e0295b0d41165029cc221.jpg  
      inflating: ./clothes_dataset/green_shirt/936d115b0f91f3267507f16b6d63a74179f6036a.jpg  
      inflating: ./clothes_dataset/green_shirt/95e492c025c4ff858d3979b1ee7a63c1fc301e24.jpg  
      inflating: ./clothes_dataset/green_shirt/9754ac1fe838544ffec5b63a17ab0fcb46712cdf.jpg  
      inflating: ./clothes_dataset/green_shirt/989b636f5b7b69a257afb738df44e7d6cc18ae59.jpg  
      inflating: ./clothes_dataset/green_shirt/9a490379340116c5f3d179dcae4dc9f2d6e0ad9b.jpg  
      inflating: ./clothes_dataset/green_shirt/9a92a1c90ab3b743d2c22b044d09f9d16fc4d2b2.jpg  
      inflating: ./clothes_dataset/green_shirt/9b04fa28c480ee8f2435aaf07d9863b2f85ab842.jpg  
      inflating: ./clothes_dataset/green_shirt/9bc1fc615e63c06aa831f017094420ecff521c3b.jpg  
      inflating: ./clothes_dataset/green_shirt/9bd1fa773cae92ef5feea93ce7513e1e29df2875.jpg  
      inflating: ./clothes_dataset/green_shirt/9be954e7481adb5d54e50b814f59a146e38a1ff3.jpg  
      inflating: ./clothes_dataset/green_shirt/9d5a620ff4700019a9c70d66d5703c8e6bcce33a.jpg  
      inflating: ./clothes_dataset/green_shirt/9e8f12518fa0d9b01c3c67511e0ab09739861cbe.jpg  
      inflating: ./clothes_dataset/green_shirt/9ebf245fa933c13e5713a3907d4340e7f5a01f01.jpg  
      inflating: ./clothes_dataset/green_shirt/a5809575d204dca27f09fc569b94340a18a32d55.jpg  
      inflating: ./clothes_dataset/green_shirt/a5aee317b047caaedaddb092cc431b1afa742388.jpg  
      inflating: ./clothes_dataset/green_shirt/a72df1839ade263c04dad22626a40239fc665bd0.jpg  
      inflating: ./clothes_dataset/green_shirt/a79c551c16de1ef97b8ceb80c70f982d23630aea.jpg  
      inflating: ./clothes_dataset/green_shirt/a7b5c4e455340ea7ee490db9c92d6f65b9dcb690.jpg  
      inflating: ./clothes_dataset/green_shirt/a921758c464179ee7b5fbc0bc2c34ce2dc74c6f5.jpg  
      inflating: ./clothes_dataset/green_shirt/aadf262cb278560ccf2eb171593d8f53c0d815d7.jpg  
      inflating: ./clothes_dataset/green_shirt/ab3894609861e39257402aa2529f179d6e72a3d3.jpg  
      inflating: ./clothes_dataset/green_shirt/ab4dd423ea40867a91017042461b1f6ed63013e0.jpg  
      inflating: ./clothes_dataset/green_shirt/aed21846e28c4091d5e1794ca33edd8d0da2aa3a.jpg  
      inflating: ./clothes_dataset/green_shirt/b095c3592f69464c00b0d646eeac316e1e1c0098.jpg  
      inflating: ./clothes_dataset/green_shirt/b0c1d48d815d259a5977e221534bc96c92f2dc17.jpg  
      inflating: ./clothes_dataset/green_shirt/b26daa252f57678e29c06efa634bcc1921ec70ed.jpg  
      inflating: ./clothes_dataset/green_shirt/b36beee4034d50b42256b2aeeffb97a87cf138e9.jpg  
      inflating: ./clothes_dataset/green_shirt/b55b84b4661d0b70cc960084a6fe985af47a1e91.jpg  
      inflating: ./clothes_dataset/green_shirt/b80aa63dc71487a0b6550734e8a28c8fd94150b7.jpg  
      inflating: ./clothes_dataset/green_shirt/b9bb9504c342dd5f9e26406de5273da50bebd477.jpg  
      inflating: ./clothes_dataset/green_shirt/ba1656c53628cf9993952b12d2d7f770a575bc4a.jpg  
      inflating: ./clothes_dataset/green_shirt/bd979b48986b68c7f6199cbc7d824667391f16af.jpg  
      inflating: ./clothes_dataset/green_shirt/bdd14e1b550dcf6f7bca16a21e4e3571fc3fe93e.jpg  
      inflating: ./clothes_dataset/green_shirt/be7d3724f80586f3635cda9875724dc6d9101470.jpg  
      inflating: ./clothes_dataset/green_shirt/bf0eecea2cd0e7635fa32e3dc33d2bcff39be362.jpg  
      inflating: ./clothes_dataset/green_shirt/c023492e94eca8f90e50ba76bcd5689418d8be50.jpg  
      inflating: ./clothes_dataset/green_shirt/c44d3adeca706ab8d7517df56ba272e94a3def65.jpg  
      inflating: ./clothes_dataset/green_shirt/c4fb30d355fec4c78d756f3e4e2671d973096bd6.jpg  
      inflating: ./clothes_dataset/green_shirt/c58a4534aab0807fb8712a3eff572cbcb8caf725.jpg  
      inflating: ./clothes_dataset/green_shirt/c68bb9f7cf6a0c01e404583c4f8c1a7e35ff5f2e.jpg  
      inflating: ./clothes_dataset/green_shirt/ca63e5b4db1c1465d5ad6a13c4239de91a01328a.jpg  
      inflating: ./clothes_dataset/green_shirt/cc6fc496589c926a0376cc7225320bde3ab02124.jpg  
      inflating: ./clothes_dataset/green_shirt/cd3cca0545e060daf505a426ab672f8b60daef44.jpg  
      inflating: ./clothes_dataset/green_shirt/cd78e374e698a0a6fdb1e62d9bd1f516b3f8e54e.jpg  
      inflating: ./clothes_dataset/green_shirt/ceafd48eb33c66551a5f6e4a35bf13f20e082b12.jpg  
      inflating: ./clothes_dataset/green_shirt/ceb7fd1c28d849e834b793269eda02372436a3b7.jpg  
      inflating: ./clothes_dataset/green_shirt/cf52ce58bafda3f3ce6881771a935afde573bfdb.jpg  
      inflating: ./clothes_dataset/green_shirt/d0c20532079a698810b1b57a711658202f2a8d08.jpg  
      inflating: ./clothes_dataset/green_shirt/d16f90c6cf6756a1b2288d16edcffb480aa7d14a.jpg  
      inflating: ./clothes_dataset/green_shirt/d2664f0f71fa6b4997304c84f1e36145bd6e15c6.jpg  
      inflating: ./clothes_dataset/green_shirt/d32dcecfac6d66d0514211c2a79008fe128270a9.jpg  
      inflating: ./clothes_dataset/green_shirt/d383c34ce5e29eaf0bdac25da27611f5980447ce.jpg  
      inflating: ./clothes_dataset/green_shirt/d46d1643da67d8d2273743f3cb91e2634694016a.jpg  
      inflating: ./clothes_dataset/green_shirt/d4eb9091d55ae5ad0dc34d7a42aa24a6773a9eee.jpg  
      inflating: ./clothes_dataset/green_shirt/d5d384caccf51afbc9f6394fc6bb05582885984e.jpg  
      inflating: ./clothes_dataset/green_shirt/d63cf9dda39169e9fa0e3a15b39ecc6905abc67c.jpg  
      inflating: ./clothes_dataset/green_shirt/d85dc5e575fab9a6c2fc8fabe3ce991f4fdee02b.jpg  
      inflating: ./clothes_dataset/green_shirt/d919762e7adea2fbf85c57055280626ae58f7aa6.jpg  
      inflating: ./clothes_dataset/green_shirt/d9516d14e530350bea570f8b9d06120421f24a11.jpg  
      inflating: ./clothes_dataset/green_shirt/d9b55075445793af25d4aadbbe99f15086898261.jpg  
      inflating: ./clothes_dataset/green_shirt/db3f51686399e0226603555affa0f62d7a58047a.jpg  
      inflating: ./clothes_dataset/green_shirt/db9243e49b04119d804e96a92d806c3a0b8f9cee.jpg  
      inflating: ./clothes_dataset/green_shirt/dce39aef3fbaf0bacda4e106d72e71c37cd564c1.jpg  
      inflating: ./clothes_dataset/green_shirt/ddfc85cf4e9e3f81c72eb40efee8aa30ab16c23a.jpg  
      inflating: ./clothes_dataset/green_shirt/def541a5ab4151f6cfe536834ea4a466a9ee440b.jpg  
      inflating: ./clothes_dataset/green_shirt/df27832a5fe792dc074f5f96f704570066e3b366.jpg  
      inflating: ./clothes_dataset/green_shirt/e0f2e98b6b7d0c7bb44e76ff438577288c2119c0.jpg  
      inflating: ./clothes_dataset/green_shirt/e16e6a28653f67493de5ce0b60498626d1f0df5c.jpg  
      inflating: ./clothes_dataset/green_shirt/e23e7dc0bd8714f855e59a95dac5bd6755636cde.jpg  
      inflating: ./clothes_dataset/green_shirt/e41dd672d9d3be8879052911ff2eadf1d7c7ca3e.jpg  
      inflating: ./clothes_dataset/green_shirt/e430a076eaa597fcab1f0707dc951aed8455b244.jpg  
      inflating: ./clothes_dataset/green_shirt/e45d9991211fa46ab50bf5f800c33c458891764e.jpg  
      inflating: ./clothes_dataset/green_shirt/eaef0e8af84d544b87516e028e28e497aad7a38a.jpg  
      inflating: ./clothes_dataset/green_shirt/ebf9b13bb00da55a3c6c4afe0bc1f537d05271f1.jpg  
      inflating: ./clothes_dataset/green_shirt/ec08edd7bd4b4f5164cd2d72888f4feb45b67a68.jpg  
      inflating: ./clothes_dataset/green_shirt/ec3329012671d17b78836cf14a3fa3335601cbf8.jpg  
      inflating: ./clothes_dataset/green_shirt/ec5925735a5cf9d91560e640d4489b26048c012b.jpg  
      inflating: ./clothes_dataset/green_shirt/ec5b86ae02ca3afa8645c1d088bcdb8ab7ed88f2.jpg  
      inflating: ./clothes_dataset/green_shirt/ee426b76eaf5280fd598e94e3f08e677bafc19cc.jpg  
      inflating: ./clothes_dataset/green_shirt/ee5074613bcb7b496e307715da768f2e08100eb5.jpg  
      inflating: ./clothes_dataset/green_shirt/ef6a4a6c75d42e281778e2aefbf65fe4dabff58d.jpg  
      inflating: ./clothes_dataset/green_shirt/f316599d852ed90150efd003e6b9088cbf318ad2.jpg  
      inflating: ./clothes_dataset/green_shirt/f3f14d4ef777663cdbe4e2ce8b0706f8c5ca7875.jpg  
      inflating: ./clothes_dataset/green_shirt/f4bf2e93bedde87a4fba9d986a7d9a80d815171c.jpg  
      inflating: ./clothes_dataset/green_shirt/f80f9848cde68328a4198732adf69f7a1e063f8a.jpg  
      inflating: ./clothes_dataset/green_shirt/f9862750ece3b27a1abaa81801da7416bdea346a.jpg  
      inflating: ./clothes_dataset/green_shirt/fb432f901f9dfea207bf27bbc066eb7d26980666.jpg  
      inflating: ./clothes_dataset/green_shirt/fb78ac1a518baa528b8d11c88c531e24c540b0e2.jpg  
      inflating: ./clothes_dataset/green_shirt/fc89295817fca1e7eefe45fee768da571ba56dcd.jpg  
      inflating: ./clothes_dataset/green_shirt/ff53085554b608e97615483adcbe8c6472561c44.jpg  
      inflating: ./clothes_dataset/green_shoes/0023b4341a2246681a83cc1c1c8e6c8eacd9f11a.jpg  
      inflating: ./clothes_dataset/green_shoes/0024daf7e1de4e9fda20142614a0b183fde93b17.jpg  
      inflating: ./clothes_dataset/green_shoes/002d73b499072cf3f3b08f780f09ffda353f7e9a.jpg  
      inflating: ./clothes_dataset/green_shoes/0087c5dc770f6aaa3e5cdffe1d45eed9ba5b52fe.jpg  
      inflating: ./clothes_dataset/green_shoes/00b56198345bfdc2342d66ab7931b78ba111fa03.jpg  
      inflating: ./clothes_dataset/green_shoes/00eca86dfc4620a77a45bdafe470a44437e5e9a0.jpg  
      inflating: ./clothes_dataset/green_shoes/011971282414e754fbb68117ab5dcfc37063e1c3.jpg  
      inflating: ./clothes_dataset/green_shoes/02fb2ab70d02279625de43bc22b7b6e9fa5dd43a.jpg  
      inflating: ./clothes_dataset/green_shoes/039f92c562f8b18b13daca14936217d286341171.jpg  
      inflating: ./clothes_dataset/green_shoes/03baac2216912f3d14939d7c58e6ac95d13c295a.jpg  
      inflating: ./clothes_dataset/green_shoes/04b4a0965cc7139a55fc1d02d7f99d3f1a958f4e.jpg  
      inflating: ./clothes_dataset/green_shoes/04b4e7d97ff0a0538473af25254826342c326803.jpg  
      inflating: ./clothes_dataset/green_shoes/05636d4a49e6a4f19522c4d6c9088f71cca3e449.jpg  
      inflating: ./clothes_dataset/green_shoes/05d38130d51620096e1bddef425e54231a9bb98a.jpg  
      inflating: ./clothes_dataset/green_shoes/067821c8f561d00c3d6db4d73164251c67f7521e.jpg  
      inflating: ./clothes_dataset/green_shoes/06e0ed753ff7a6f0bf2123d3fced6346072d3898.jpg  
      inflating: ./clothes_dataset/green_shoes/07a4505e23f5a03715ba2ce550890cedda896ea1.jpg  
      inflating: ./clothes_dataset/green_shoes/083d5cdf7608f43710d6d4ec0251f9873cbcdf99.jpg  
      inflating: ./clothes_dataset/green_shoes/095ce44f7197c1b3d7f9a6fee6cd8193bafd0ff1.jpg  
      inflating: ./clothes_dataset/green_shoes/0a2ac12795025e8734b1ae8fafe0abb1361e9738.jpg  
      inflating: ./clothes_dataset/green_shoes/0a6d29d8f08a5b6409d0e28858eb8f23411e9f20.jpg  
      inflating: ./clothes_dataset/green_shoes/0abc51f18c8fa3991fb34cefda0273de64c651ad.jpg  
      inflating: ./clothes_dataset/green_shoes/0ba434ec362807390a5813b565c1e14042c34e86.jpg  
      inflating: ./clothes_dataset/green_shoes/0bcba6d8ea258b74f4e0a980deddc382b0a329f2.jpg  
      inflating: ./clothes_dataset/green_shoes/0bd6fef83d80e265cdfdaad41c9eb90e00e48fb0.jpg  
      inflating: ./clothes_dataset/green_shoes/0be1391b2ed0412aedf6c5f9a49a25ea19d426f1.jpg  
      inflating: ./clothes_dataset/green_shoes/0bff83e85613644c88f0d485910473190ab52ae1.jpg  
      inflating: ./clothes_dataset/green_shoes/0c04aa86071eccd945dd04a7aedba4a345ab14c4.jpg  
      inflating: ./clothes_dataset/green_shoes/0c71b9515863faf896c2cebacaaa18553f11f7a0.jpg  
      inflating: ./clothes_dataset/green_shoes/0c998dbddc2d666529836ae11e81c861c0f4e6ac.jpg  
      inflating: ./clothes_dataset/green_shoes/0dd1621f366d57bce88681d079c8f42c6e2d890c.jpg  
      inflating: ./clothes_dataset/green_shoes/0df23f0f357a9afd0242e0d280ec36c19459a112.jpg  
      inflating: ./clothes_dataset/green_shoes/0f00784e9b487f7c6f7491f7950572f6610d3376.jpg  
      inflating: ./clothes_dataset/green_shoes/0f0737cab94302cfaad27f07b1256ccef1ccef91.jpg  
      inflating: ./clothes_dataset/green_shoes/0f381976e8ec70d7b61efb05e3046bff184fb66d.jpg  
      inflating: ./clothes_dataset/green_shoes/1111b6022b1a5fad62f63797276243f525ec941d.jpg  
      inflating: ./clothes_dataset/green_shoes/11153b902019ece3cbebfe8e21ae35b6408346e6.jpg  
      inflating: ./clothes_dataset/green_shoes/12587d73088c732bdf4f8f6432b8d0093de2514a.jpg  
      inflating: ./clothes_dataset/green_shoes/12b23c04d923070565751896d521af4ba0c0e298.jpg  
      inflating: ./clothes_dataset/green_shoes/12e1349faa005dac1a232e545a52de1acda964c7.jpg  
      inflating: ./clothes_dataset/green_shoes/139b938fe1dbbd9911ffae8e75550a2100047e3c.jpg  
      inflating: ./clothes_dataset/green_shoes/1421a588299db94af7549d879c7f567270fa3cba.jpg  
      inflating: ./clothes_dataset/green_shoes/1452b050130f52a3753bbb3fba169c2d5a599b5a.jpg  
      inflating: ./clothes_dataset/green_shoes/1467befa2aa72cb3853175f82823ee710e9ef10b.jpg  
      inflating: ./clothes_dataset/green_shoes/14e041cfd430aae48871c28a891f7ce519e4d677.jpg  
      inflating: ./clothes_dataset/green_shoes/14f7dc5a556f04a140652ecb018dbf18cca33eb0.jpg  
      inflating: ./clothes_dataset/green_shoes/153b144b0e1b478a7be9fe8f8aa854e820a53cf8.jpg  
      inflating: ./clothes_dataset/green_shoes/15436ae50d5cbe9538f446599b25d6cee4624ca4.jpg  
      inflating: ./clothes_dataset/green_shoes/167d869aa66ffb70ea6d63bdfe3324080aa6f739.jpg  
      inflating: ./clothes_dataset/green_shoes/169e4d396e24a5db9dae46cff6f74cb0c92a7a03.jpg  
      inflating: ./clothes_dataset/green_shoes/177310972ad6bec43a71ba3ac69693278c0e61e2.jpg  
      inflating: ./clothes_dataset/green_shoes/17f0f3006e222a7f3a7a2e054e0c2c6d096419cc.jpg  
      inflating: ./clothes_dataset/green_shoes/1865c7d4a62b771c71ad1cca239024d9d85f0794.jpg  
      inflating: ./clothes_dataset/green_shoes/186a136553e0134f9fe5d568f1a68ca1c0d034a5.jpg  
      inflating: ./clothes_dataset/green_shoes/1afaa580c580da10279a2cf376c5dbf387b729c2.jpg  
      inflating: ./clothes_dataset/green_shoes/1b1b69fea24e0aecae2343ea4368cbbd7c55e7a0.jpg  
      inflating: ./clothes_dataset/green_shoes/1b5d5e2ac58e9e46fd33aac951a85f5c0add3c72.jpg  
      inflating: ./clothes_dataset/green_shoes/1b7c2e3dac388e09aaa09a68a3884fb371b0bf24.jpg  
      inflating: ./clothes_dataset/green_shoes/1bf75638388beb77815b3217f7cda263c5055356.jpg  
      inflating: ./clothes_dataset/green_shoes/1c51506b4d53246491a2ade06da2fa12043716ef.jpg  
      inflating: ./clothes_dataset/green_shoes/1c5f4e6baf01bfe3db820391bbdcf886cb6405b6.jpg  
      inflating: ./clothes_dataset/green_shoes/1ce06eec794a1a9a0af580bff25dea2f87b25a95.jpg  
      inflating: ./clothes_dataset/green_shoes/1d33909f638786efb17f4e48c183b4bcb15b84af.jpg  
      inflating: ./clothes_dataset/green_shoes/1d5a4f2c3171f5e42b7bd4c06aa5712ad3de1d32.jpg  
      inflating: ./clothes_dataset/green_shoes/1d5de865d0c4c0a8e0864079f60a511cfb050b79.jpg  
      inflating: ./clothes_dataset/green_shoes/1d91697e349bd8f6a9b697fe4d4bfcf5fc547f7c.jpg  
      inflating: ./clothes_dataset/green_shoes/1d93f816fc4216e1716c02a314355bf988b0975b.jpg  
      inflating: ./clothes_dataset/green_shoes/1e1015858557245a8d68e6d21950ed5138bb1677.jpg  
      inflating: ./clothes_dataset/green_shoes/1e109e2151bd16b060f970cfc81c115b1fcadce9.jpg  
      inflating: ./clothes_dataset/green_shoes/1e5eaacc3731a40fc1b2a1036ef52a46fd68e02e.jpg  
      inflating: ./clothes_dataset/green_shoes/1e78229b1ea6983b36851be79e724d803793abb7.jpg  
      inflating: ./clothes_dataset/green_shoes/1e822c6025fc71b5cbb0456330ba12782f8379e9.jpg  
      inflating: ./clothes_dataset/green_shoes/1ffc67ad1c18e9ee14acf550d4553fc12d92aee8.jpg  
      inflating: ./clothes_dataset/green_shoes/1ffd0b09af76850055ebbf637a249bef9ac607b3.jpg  
      inflating: ./clothes_dataset/green_shoes/207163558537f9a5ab555383b0939363579ebda9.jpg  
      inflating: ./clothes_dataset/green_shoes/208cb1c59b83fc73e3a5f8a76196b89bd89026a5.jpg  
      inflating: ./clothes_dataset/green_shoes/215f217355d30c67efc6e42ef71261356ad8a79c.jpg  
      inflating: ./clothes_dataset/green_shoes/21b502b03ce298ea67b8e5ff2cae61fca0500ddb.jpg  
      inflating: ./clothes_dataset/green_shoes/2266e2830e5cbbcf844ca38cff9dcdab3b7846d1.jpg  
      inflating: ./clothes_dataset/green_shoes/22a0b1e483a833f0e9c43bcf204820dcff09b109.jpg  
      inflating: ./clothes_dataset/green_shoes/22b2c7125e8215f6e6f60b4f281dd37082c3da62.jpg  
      inflating: ./clothes_dataset/green_shoes/22df46bc224751667ed8e950238a785f9febd510.jpg  
      inflating: ./clothes_dataset/green_shoes/230ae165a1ddcf08912738d5dc1a867f95f6b5ff.jpg  
      inflating: ./clothes_dataset/green_shoes/2345068e6e6dc344c5094f43f8934118a39f0eb8.jpg  
      inflating: ./clothes_dataset/green_shoes/24b26410a82c111b78dcf6f59cd3bd4c6cc80c3c.jpg  
      inflating: ./clothes_dataset/green_shoes/24f89ef8cac2b00da3a6cea79806c3da9e0f6f77.jpg  
      inflating: ./clothes_dataset/green_shoes/27adab99f8980e828c11d9111f95b4a0ee87cf76.jpg  
      inflating: ./clothes_dataset/green_shoes/27cbc09365481a288aeaae4b0fc810ce568d5954.jpg  
      inflating: ./clothes_dataset/green_shoes/28f9a9cc9bb06be1e311707aad75d63ff478a12f.jpg  
      inflating: ./clothes_dataset/green_shoes/29e994aec48f78b67fe67ee870a07388e4455f95.jpg  
      inflating: ./clothes_dataset/green_shoes/2ae6978cacca1ef8e53cb135bf8ec558948f76ed.jpg  
      inflating: ./clothes_dataset/green_shoes/2b040f4c1da5601bc74454a84c4df877f2a8ad8e.jpg  
      inflating: ./clothes_dataset/green_shoes/2b14cf1e8e070e8d53a536156dc8828d0f1d131e.jpg  
      inflating: ./clothes_dataset/green_shoes/2b1f0640891c8a3bd4c24ff0ee1c1ab091ec02d5.jpg  
      inflating: ./clothes_dataset/green_shoes/2b53daa0e8aac16fbf28ec0df043f5f330800713.jpg  
      inflating: ./clothes_dataset/green_shoes/2b5d795ec88fd64716c8b38004837d739462149e.jpg  
      inflating: ./clothes_dataset/green_shoes/2b8da145b8fbfd40d29b4c8bc30f30e4b14617ed.jpg  
      inflating: ./clothes_dataset/green_shoes/2bc0dfe62a5e2609250b78f52cee238d3d48904c.jpg  
      inflating: ./clothes_dataset/green_shoes/2dcfdf63bf5843d894f6037fa8822db1108d61fa.jpg  
      inflating: ./clothes_dataset/green_shoes/2e124d2ac4b35768696b5b3db44ea8bcc4e5e92b.jpg  
      inflating: ./clothes_dataset/green_shoes/2e71422aa30b05a9a8d51d531611a65fef88f862.jpg  
      inflating: ./clothes_dataset/green_shoes/2eb41c1c72c48083080b89536a42c6adb8ba47f8.jpg  
      inflating: ./clothes_dataset/green_shoes/2f5da7bad9a4410cd0e87588fd11d2a0c90ebe56.jpg  
      inflating: ./clothes_dataset/green_shoes/2f874303ebf604253a0a98106c2c35c70c5551c6.jpg  
      inflating: ./clothes_dataset/green_shoes/2fc061dc4ba9d0dfdfffb5bec92678e33e0857b0.jpg  
      inflating: ./clothes_dataset/green_shoes/2fc693e6b08cf615571396c4ae4b6961cf47cdbb.jpg  
      inflating: ./clothes_dataset/green_shoes/310fb7577bfe690f9458aa673d7bcb1af02e90be.jpg  
      inflating: ./clothes_dataset/green_shoes/317a9216c7c3feb7f35ae113bf28b8a9c3c14865.jpg  
      inflating: ./clothes_dataset/green_shoes/31aa8a98611b0fb0330a720b638fd5d69d41fa7c.jpg  
      inflating: ./clothes_dataset/green_shoes/31b66185a591d2ecd21e33b9e4765460170582f6.jpg  
      inflating: ./clothes_dataset/green_shoes/32529e295222e468d222e5e9261c9bdfeb331c20.jpg  
      inflating: ./clothes_dataset/green_shoes/3286793fa1f3d1ad2d68a9171ddb73c4041a8624.jpg  
      inflating: ./clothes_dataset/green_shoes/33d7f50e69537a4338c0059942487be2474aba57.jpg  
      inflating: ./clothes_dataset/green_shoes/357c6f1a3dc834dcf18ac486fb64e04da219d2f6.jpg  
      inflating: ./clothes_dataset/green_shoes/3581e26c8e37e6b3ecbabdf6991bdad3661753e1.jpg  
      inflating: ./clothes_dataset/green_shoes/36d2b0e8647afbcfbc2e3576e32d631f31cb9f35.jpg  
      inflating: ./clothes_dataset/green_shoes/36d803b173cd921ead4c2b24eb97c4486c425bdf.jpg  
      inflating: ./clothes_dataset/green_shoes/3833b0cf39dd1127f1c9ceaf45ba26bb069b4b01.jpg  
      inflating: ./clothes_dataset/green_shoes/38c60ee8d8594d900f7d17749339642e6f83e21a.jpg  
      inflating: ./clothes_dataset/green_shoes/38ee939cb3db2a7f81b8e63ea385b8d366433b0d.jpg  
      inflating: ./clothes_dataset/green_shoes/395068069fa9c904a134af37145b369375b8994c.jpg  
      inflating: ./clothes_dataset/green_shoes/3a435c94266417ed6e5d259e97f2c8c0700db00a.jpg  
      inflating: ./clothes_dataset/green_shoes/3a5574e9cca12c381b4c4ef351d74458d804069e.jpg  
      inflating: ./clothes_dataset/green_shoes/3a6e05454f4ffc31badcb5770207a3bec2b48118.jpg  
      inflating: ./clothes_dataset/green_shoes/3ac534b511ea5aba1370c6279d36386d324410c5.jpg  
      inflating: ./clothes_dataset/green_shoes/3ac587ed659f7380191582ddebda832ddd7bdc03.jpg  
      inflating: ./clothes_dataset/green_shoes/3ad8d9a7e728996e46470b9b8c52655bff7dbac7.jpg  
      inflating: ./clothes_dataset/green_shoes/3b9b361295f947ec76d7729d92ea41769059efe4.jpg  
      inflating: ./clothes_dataset/green_shoes/3c1a0ffde62e458e99c04953abceba5c024cae83.jpg  
      inflating: ./clothes_dataset/green_shoes/3c9322185f56e8985a9dac61de94f805bfed5565.jpg  
      inflating: ./clothes_dataset/green_shoes/3d032473555a6fa37e5568e83f84f4c94bbc0c8b.jpg  
      inflating: ./clothes_dataset/green_shoes/3d10e800da8227a6ecb7a164d6d8eb4beebb1942.jpg  
      inflating: ./clothes_dataset/green_shoes/3d6e6b92c53cd04e1ff0acb3cd70b2f707c6bda7.jpg  
      inflating: ./clothes_dataset/green_shoes/3dd0557a60ffa71a9a622217c3fbd135518d6e6d.jpg  
      inflating: ./clothes_dataset/green_shoes/3dd2ae3fed0ded3cd994028141af3c34885f110d.jpg  
      inflating: ./clothes_dataset/green_shoes/3e2ba97d9f1924c3493ba99f3eb638f95ad818a3.jpg  
      inflating: ./clothes_dataset/green_shoes/3fb436b603da2094831379d218d79ac94696c008.jpg  
      inflating: ./clothes_dataset/green_shoes/406978adce2f236039617b1905676d744fb44d1b.jpg  
      inflating: ./clothes_dataset/green_shoes/417521b1c8875f0e418d1cb1182f8d64e89046cf.jpg  
      inflating: ./clothes_dataset/green_shoes/41999485c5b9f52619e14c4b105e3ebd6ac39672.jpg  
      inflating: ./clothes_dataset/green_shoes/41a71e44282a83017eb272c9fcc97442750ece8a.jpg  
      inflating: ./clothes_dataset/green_shoes/41e2aeafa166eaeb4718e3bc0ab26b5e755785d7.jpg  
      inflating: ./clothes_dataset/green_shoes/43880a794fe55c541c22453218449e5dfa26bbf3.jpg  
      inflating: ./clothes_dataset/green_shoes/44b1de484cf17f9684e226f78a6cc29b325d20a1.jpg  
      inflating: ./clothes_dataset/green_shoes/450c99bda54fe474825361f0cc70bbad78f49b47.jpg  
      inflating: ./clothes_dataset/green_shoes/457ad971bed122438e6a0124b5357c35253cb065.jpg  
      inflating: ./clothes_dataset/green_shoes/45f3c67634756fced844a73ab26f9b0855fa3166.jpg  
      inflating: ./clothes_dataset/green_shoes/46ef4d4a33c8e8849c26b760ded19e4d04c91f02.jpg  
      inflating: ./clothes_dataset/green_shoes/473f7d036ac47644792f5642856a5e07bd933c5b.jpg  
      inflating: ./clothes_dataset/green_shoes/4813fb42c825d43a33fcff8bfe568de67965182f.jpg  
      inflating: ./clothes_dataset/green_shoes/48af2b5985849623665c4b1d7d7f393e437cd54d.jpg  
      inflating: ./clothes_dataset/green_shoes/48cc304318f7f93617ccf4cb04659731110e25ac.jpg  
      inflating: ./clothes_dataset/green_shoes/494ab867291318f0fdd4a3550672ecab0c94b19f.jpg  
      inflating: ./clothes_dataset/green_shoes/49a554702504347d786fa3f793b23151f02e7759.jpg  
      inflating: ./clothes_dataset/green_shoes/4a3cc213314e150b179a03aa33d282675218c387.jpg  
      inflating: ./clothes_dataset/green_shoes/4b09127616f1646ed669654b0ea3f656bd3df946.jpg  
      inflating: ./clothes_dataset/green_shoes/4b7c736574c4d14be727fe719789f59da91df469.jpg  
      inflating: ./clothes_dataset/green_shoes/4bf4fb9c83614cdf2f8ecca2b776fdd7fd3abf74.jpg  
      inflating: ./clothes_dataset/green_shoes/4d1dd0ef4a11373362ada7ded3c148dcc6011f62.jpg  
      inflating: ./clothes_dataset/green_shoes/4e4de99fc94cd6c0566da2f05a7a0b1b0752906a.jpg  
      inflating: ./clothes_dataset/green_shoes/4f24444d9ad8da1506cb09ce93d8c521a8ab0573.jpg  
      inflating: ./clothes_dataset/green_shoes/4f2aba02e529c677125e71620ce3acd102f831e7.jpg  
      inflating: ./clothes_dataset/green_shoes/500051d9b8cedae32e9a517160b74d6db43003ce.jpg  
      inflating: ./clothes_dataset/green_shoes/5007aa457293502de9eee2a35eed8bd99cb978b8.jpg  
      inflating: ./clothes_dataset/green_shoes/50777c0dafd5b51dcd956790166ffbbb85f7b271.jpg  
      inflating: ./clothes_dataset/green_shoes/508071692fcaf4c2012f2cb457bce40bd8dc988f.jpg  
      inflating: ./clothes_dataset/green_shoes/51464cd7fdb9ca8830604ade3bfc458cd92c5932.jpg  
      inflating: ./clothes_dataset/green_shoes/51edabf2eafed0c10a1b833d70977a0890607183.jpg  
      inflating: ./clothes_dataset/green_shoes/524c15c5de1b240ad2ff4fffb36a6ca73bf549ce.jpg  
      inflating: ./clothes_dataset/green_shoes/525c020c1a8c21d201de98051acd33bae01b2500.jpg  
      inflating: ./clothes_dataset/green_shoes/5292ed372e89718b92684368fd9cef911ef6c8c7.jpg  
      inflating: ./clothes_dataset/green_shoes/52c6c1488f4d4b3cbb610ddfa595f3cf67fea905.jpg  
      inflating: ./clothes_dataset/green_shoes/55ef06fa770266e05b839a942e547fe156af47e8.jpg  
      inflating: ./clothes_dataset/green_shoes/56788ef3f191db613bc0d5be2d62aa8c8bfcc476.jpg  
      inflating: ./clothes_dataset/green_shoes/572f401b96160c4da49449a029ffb67cf8c150e5.jpg  
      inflating: ./clothes_dataset/green_shoes/577488adb6760c1a929f520ee6a0e65cfed19a60.jpg  
      inflating: ./clothes_dataset/green_shoes/5832fee963329db2a4e5c4567ad579a55d26fc4d.jpg  
      inflating: ./clothes_dataset/green_shoes/597e70c25679f8c5f27dea6acbb3966acf6cf5be.jpg  
      inflating: ./clothes_dataset/green_shoes/5a74f5501475a7ad910ff4b909509d301d3a94da.jpg  
      inflating: ./clothes_dataset/green_shoes/5b5f5e5e97c303633322a1f97cda861fb0abe642.jpg  
      inflating: ./clothes_dataset/green_shoes/5bd4f20b8c86792e74eea25e2f8734b1da47e53f.jpg  
      inflating: ./clothes_dataset/green_shoes/5bf0d1d97b758cd24e82fb1accafb553d90582af.jpg  
      inflating: ./clothes_dataset/green_shoes/5c628c81294731cfa47c53b43a7d6f4f764d0d3d.jpg  
      inflating: ./clothes_dataset/green_shoes/5cfca5d817fc895a5032136931b4b42a7c03b467.jpg  
      inflating: ./clothes_dataset/green_shoes/5d4a713cea478bc93aef2c922840ccb397414c5e.jpg  
      inflating: ./clothes_dataset/green_shoes/5d608459b366c14c5ef6dbb74e9273a4da59938b.jpg  
      inflating: ./clothes_dataset/green_shoes/5e18d698b618eb83b2f8cd9656c0e1d37667ce35.jpg  
      inflating: ./clothes_dataset/green_shoes/5f82e16cf0e1292834298f32eeefaaa847639f25.jpg  
      inflating: ./clothes_dataset/green_shoes/6151e73fea817d1cf49901b5760356f3b60ff199.jpg  
      inflating: ./clothes_dataset/green_shoes/61b9f7ed89882700f377a513c71a9e07f39dfe03.jpg  
      inflating: ./clothes_dataset/green_shoes/61e0a0d58a4e5155c323f9ad1e5888bd3d6aee5c.jpg  
      inflating: ./clothes_dataset/green_shoes/62d1250fe37318783ce23b594775a611b71dcefe.jpg  
      inflating: ./clothes_dataset/green_shoes/645e80742787df0710ff1957c0db017369a9bae5.jpg  
      inflating: ./clothes_dataset/green_shoes/645e9d8fcfe29fbe3129be06c271459ee642ddaa.jpg  
      inflating: ./clothes_dataset/green_shoes/658aa9726d2cc6a82a24efb2cd47c0a7b48c00f3.jpg  
      inflating: ./clothes_dataset/green_shoes/65d9151c266cb8d1f22224b4e99dfac6e09e8f0d.jpg  
      inflating: ./clothes_dataset/green_shoes/65de0d5c03bd621daba439c6f1ef9a3778650252.jpg  
      inflating: ./clothes_dataset/green_shoes/684f447939c672b8194da28a3219983ea4c93c08.jpg  
      inflating: ./clothes_dataset/green_shoes/685c8783c1ebf7565780f8a12a01a22e759e2f7c.jpg  
      inflating: ./clothes_dataset/green_shoes/68e060cfe4bda8b634ab25bda0b2dc9d97f8b1d3.jpg  
      inflating: ./clothes_dataset/green_shoes/6a016bb15a5bb409d892975a2a8f3f33cbe1ef2d.jpg  
      inflating: ./clothes_dataset/green_shoes/6b4a6ecd906b2860cffa82141b195e745bb12bf4.jpg  
      inflating: ./clothes_dataset/green_shoes/6b6d1a23ad7a0f4c0eca5512a324809b31afbe0c.jpg  
      inflating: ./clothes_dataset/green_shoes/6ce15575bcfba53b17c37633daefbe3fca731c28.jpg  
      inflating: ./clothes_dataset/green_shoes/6d1bb446e27f71a179bc2d5545940098d4861d7c.jpg  
      inflating: ./clothes_dataset/green_shoes/6e425136c3a993ebc51186772019888f4c8250d7.jpg  
      inflating: ./clothes_dataset/green_shoes/6e48aecde901a0b42158a882aa7a0dee417d137b.jpg  
      inflating: ./clothes_dataset/green_shoes/6e6837704ca0758f67affcbe835e1a397d1c5bc0.jpg  
      inflating: ./clothes_dataset/green_shoes/6e99831781a1436c36cb35533afdd415191a2f86.jpg  
      inflating: ./clothes_dataset/green_shoes/702cbe1af76fed14cb6712f5bc13b1c7afc5c22d.jpg  
      inflating: ./clothes_dataset/green_shoes/7054d2d8355c0a1773cfd7c93796e666fe4ba470.jpg  
      inflating: ./clothes_dataset/green_shoes/7078d5a7d4e01355e0786ec6174ee4f27534bcfc.jpg  
      inflating: ./clothes_dataset/green_shoes/7106e4a9c14b137485984a7ef7edddc38045ae43.jpg  
      inflating: ./clothes_dataset/green_shoes/7114f6fbb0526f448cf453b1d25aff84d86ea8f0.jpg  
      inflating: ./clothes_dataset/green_shoes/71233ca27db218dd4d8acfde1d78556d3e8965c9.jpg  
      inflating: ./clothes_dataset/green_shoes/71e0550575fd71bc932eb9ed3b944b88301a0c69.jpg  
      inflating: ./clothes_dataset/green_shoes/72704947f9be338d80962daf930e27c50cc5a67a.jpg  
      inflating: ./clothes_dataset/green_shoes/7286483053d9b57fa22ded91c25c810fb9047a30.jpg  
      inflating: ./clothes_dataset/green_shoes/736eb5e9534247c00c4566f3e13d521af5b6fd30.jpg  
      inflating: ./clothes_dataset/green_shoes/7424ac67b0e7ef698ae7b2acae248bab1496b45d.jpg  
      inflating: ./clothes_dataset/green_shoes/74e8f94207a75653fb67567dc7123d7255eaf44c.jpg  
      inflating: ./clothes_dataset/green_shoes/7521708023d7b7fd7acf5261157fa7a5c0ad330f.jpg  
      inflating: ./clothes_dataset/green_shoes/7559448c3e5298397b3959828d236b2aa03f6085.jpg  
      inflating: ./clothes_dataset/green_shoes/75604b600661a3cc11cf9485e7c7441b4d4c4bf7.jpg  
      inflating: ./clothes_dataset/green_shoes/761550f1db504b3ab6015312ab50b1f54f90cbf9.jpg  
      inflating: ./clothes_dataset/green_shoes/76e3ee91759204b5b7c312a38e266f08d913080e.jpg  
      inflating: ./clothes_dataset/green_shoes/781546d449935e32fdb785272db6b82d6e82c651.jpg  
      inflating: ./clothes_dataset/green_shoes/7941ba63962feefc2d60e6c540e57dba4beef1eb.jpg  
      inflating: ./clothes_dataset/green_shoes/799f9d6e77b96b2332eb41f1daed45cbc29d42b6.jpg  
      inflating: ./clothes_dataset/green_shoes/79b3cbb51b03d7eff760565ea34e80f5aca0f686.jpg  
      inflating: ./clothes_dataset/green_shoes/7a303ffb26f6324d6639b2c7db36d0d1cc145d45.jpg  
      inflating: ./clothes_dataset/green_shoes/7a76911b07fb6ae29d621cd3dd7e1431a4c21c3a.jpg  
      inflating: ./clothes_dataset/green_shoes/7be64fdb056031d867868bee31bec967ba87c745.jpg  
      inflating: ./clothes_dataset/green_shoes/7d1053b7d359293cd501f52be82dc112af5ec6d2.jpg  
      inflating: ./clothes_dataset/green_shoes/7ea3acd9b8290c4f414414e223f1c39df66ad590.jpg  
      inflating: ./clothes_dataset/green_shoes/7eee433aa3c629360c1401dd3f45edd3f2d427b6.jpg  
      inflating: ./clothes_dataset/green_shoes/7fff7500f634cf0f01b117f6307eacaef5a5e2e6.jpg  
      inflating: ./clothes_dataset/green_shoes/80cfd6d45f842a23f3b876061ed50f72316f84b6.jpg  
      inflating: ./clothes_dataset/green_shoes/8194967df0e1f513fcf1fa7b2b747548e3805968.jpg  
      inflating: ./clothes_dataset/green_shoes/81cf2e5c63fd8513c6e3681984649bdb62f76319.jpg  
      inflating: ./clothes_dataset/green_shoes/829be154a8cbd150b8d3c28d316b3587aca08b4f.jpg  
      inflating: ./clothes_dataset/green_shoes/82d684ad570f1fc4785be6752c23b517ac335291.jpg  
      inflating: ./clothes_dataset/green_shoes/831f84b855bf595439f279d0a1162993b0df4717.jpg  
      inflating: ./clothes_dataset/green_shoes/835f6f9057b1d55567b7dfd04584886be1a4cccc.jpg  
      inflating: ./clothes_dataset/green_shoes/83736971f4e7e75156f72e37f3bb855240e36186.jpg  
      inflating: ./clothes_dataset/green_shoes/847519ca689319638c654f0c50c34b84c1cb7de7.jpg  
      inflating: ./clothes_dataset/green_shoes/84a8a34eacce80d2c857566d4785a0e569d46d4b.jpg  
      inflating: ./clothes_dataset/green_shoes/84ac88298112c41c5644dfcac6a05250c93ed302.jpg  
      inflating: ./clothes_dataset/green_shoes/86b100e607c02a0b27d767c31d450fd5aa29b568.jpg  
      inflating: ./clothes_dataset/green_shoes/876315068969b10151b33eadb2a87334faffe1a4.jpg  
      inflating: ./clothes_dataset/green_shoes/877bb3ee71aa3654b04f37ad7c86b79ddf742069.jpg  
      inflating: ./clothes_dataset/green_shoes/87d6b696bd49aa2f8d453943ebe3e7696898347b.jpg  
      inflating: ./clothes_dataset/green_shoes/881cca66c088a24483d7f2cd250bee62ef03f2a6.jpg  
      inflating: ./clothes_dataset/green_shoes/88406f03df4fe34e606601862c68246dab28ada8.jpg  
      inflating: ./clothes_dataset/green_shoes/88c49e220fc68171ed132eccb381aa389e7b4fd9.jpg  
      inflating: ./clothes_dataset/green_shoes/898739c4ce6a54c5f56c71862e592e60a8e12741.jpg  
      inflating: ./clothes_dataset/green_shoes/8b1a5c68e45f399cfa0231f4219585c5af83374d.jpg  
      inflating: ./clothes_dataset/green_shoes/8b1b99dff54227f63b46ae5cdf5d86b3d38b9245.jpg  
      inflating: ./clothes_dataset/green_shoes/8baa9e0669c20bb270179583d6b33cd376cdb560.jpg  
      inflating: ./clothes_dataset/green_shoes/8c387eaea35d42a728685c985e9bfbe952c22ac7.jpg  
      inflating: ./clothes_dataset/green_shoes/8c38a91520a2eff11e325c4da48d4ed860cf8e56.jpg  
      inflating: ./clothes_dataset/green_shoes/8d0ce20e29949e69f021904c3221e0ed26c29abf.jpg  
      inflating: ./clothes_dataset/green_shoes/8d56406c94a47ce4877d8c0dc79a31b810108f99.jpg  
      inflating: ./clothes_dataset/green_shoes/8d6b022580f6b5f874a02eff716a02f7b190ce75.jpg  
      inflating: ./clothes_dataset/green_shoes/8d8c56b98beb3be8d8939903d401b103c8bb1cc0.jpg  
      inflating: ./clothes_dataset/green_shoes/8eeb655ae81d1c24f8d754eabbc6905d56458100.jpg  
      inflating: ./clothes_dataset/green_shoes/8f3e4263ee7f127647cdc8edb457a0c29c6e4552.jpg  
      inflating: ./clothes_dataset/green_shoes/8fcaa7abfdf244050827c8b1845a56c5fc972728.jpg  
      inflating: ./clothes_dataset/green_shoes/8fe7d90836f96f60fa10333e2931819691c92190.jpg  
      inflating: ./clothes_dataset/green_shoes/9093b98872f5f6aee2042b5e88deb2444189ec39.jpg  
      inflating: ./clothes_dataset/green_shoes/90fc18f9301a71612a44344c4248bf155076a75d.jpg  
      inflating: ./clothes_dataset/green_shoes/9261aa95ed05ca27be6b97d49c225ca48c908611.jpg  
      inflating: ./clothes_dataset/green_shoes/929f4d569d1faf2dabf61269c719f756f78de833.jpg  
      inflating: ./clothes_dataset/green_shoes/92b4adc006cb6592a46c0b69c0059f1a3c5534cd.jpg  
      inflating: ./clothes_dataset/green_shoes/945dfc17dad2c86e21d9cabea00c4a0bcecd785a.jpg  
      inflating: ./clothes_dataset/green_shoes/94e738bdd26a595c47469a7e15ac3d6d02c125b9.jpg  
      inflating: ./clothes_dataset/green_shoes/94ee6c4df366d894e479d9344af63edab8115501.jpg  
      inflating: ./clothes_dataset/green_shoes/96974e51403f68b0249f8da5a806c9d0d556b572.jpg  
      inflating: ./clothes_dataset/green_shoes/98b4dd714cafe11b113c9a85b8f514ffe49dee02.jpg  
      inflating: ./clothes_dataset/green_shoes/98e44c6f52051128cb77bc96a2e2ec285de120e9.jpg  
      inflating: ./clothes_dataset/green_shoes/9acf50152065267845a3ef1c6824ccc5f277cb4f.jpg  
      inflating: ./clothes_dataset/green_shoes/9bfc20845b71d582194aaf3fd8544591fe51d2f7.jpg  
      inflating: ./clothes_dataset/green_shoes/9c0b37927b3c60e7d5fbbccaf8e758956da6a06f.jpg  
      inflating: ./clothes_dataset/green_shoes/9cdcd77ba0b0106b85fa3bf4f086dfdab17fe0b1.jpg  
      inflating: ./clothes_dataset/green_shoes/9d6cc6c65103bebfedc98354eed2d780b1cfc737.jpg  
      inflating: ./clothes_dataset/green_shoes/9d78735f7259c5975b5c26fc724716a528cbfc30.jpg  
      inflating: ./clothes_dataset/green_shoes/9e1990d5988f50205c872c44a244d1c019ac7fd1.jpg  
      inflating: ./clothes_dataset/green_shoes/9e53f98858187372f7374d23f56150f5ba817649.jpg  
      inflating: ./clothes_dataset/green_shoes/9f94840b25e649dbc07ac1f64e96925bf11c5c0a.jpg  
      inflating: ./clothes_dataset/green_shoes/a000ae9b92a0612411c1863925edc7c879846ba1.jpg  
      inflating: ./clothes_dataset/green_shoes/a26b9b8d97fe2475010465ca4ce411ce8d21d9a1.jpg  
      inflating: ./clothes_dataset/green_shoes/a2795d646dc8b0755162a91ceff191d44c555353.jpg  
      inflating: ./clothes_dataset/green_shoes/a2d09f850b696a3f617221b353b7979313fcb3ae.jpg  
      inflating: ./clothes_dataset/green_shoes/a32e468c603b2764c21ae1db8637277729b8c77f.jpg  
      inflating: ./clothes_dataset/green_shoes/a3469fdab51da90920fd2ea184b6634b9741b2fd.jpg  
      inflating: ./clothes_dataset/green_shoes/a39515ceaf0f73c53bdb2cd50ab3edf461337fe5.jpg  
      inflating: ./clothes_dataset/green_shoes/a3a6a424b8f607727cde4675d784ecb51d69b071.jpg  
      inflating: ./clothes_dataset/green_shoes/a3c6be52f4d4d000d78c8655fe1b22e709813537.jpg  
      inflating: ./clothes_dataset/green_shoes/a4d76e6cc7f2243a96728225e1f4a27d36410738.jpg  
      inflating: ./clothes_dataset/green_shoes/a4e155f7249a591e35d6cb16cfef1d9ee07b0e44.jpg  
      inflating: ./clothes_dataset/green_shoes/a51077b811bf848bb90aaabb465afc1ca5f998b4.jpg  
      inflating: ./clothes_dataset/green_shoes/a531e7bf4049518587b1a6ef297b968c659ebbe7.jpg  
      inflating: ./clothes_dataset/green_shoes/a5ae1c1c09222c629e68139e2731d35944a28096.jpg  
      inflating: ./clothes_dataset/green_shoes/a803e1bf6c7c61372936bc2985e67d4acf7b3297.jpg  
      inflating: ./clothes_dataset/green_shoes/a81f2ce2808dd307cc53467abc01e8535bc4bee6.jpg  
      inflating: ./clothes_dataset/green_shoes/a88c97ac3a6640ce8326953b703c9ed37189d7f7.jpg  
      inflating: ./clothes_dataset/green_shoes/aa15590a9725e10701e028d77e6a5fa594d45dc9.jpg  
      inflating: ./clothes_dataset/green_shoes/aa920f50e142c99c253e1cdf890fbc27b49ae544.jpg  
      inflating: ./clothes_dataset/green_shoes/aaeba68f98a07bf7638726013a978d9fd531e951.jpg  
      inflating: ./clothes_dataset/green_shoes/ab845fa6ad1c67e6e281bc4bc58da395c3c2f48c.jpg  
      inflating: ./clothes_dataset/green_shoes/abd5d4c216882e00f370348a07bc429528742252.jpg  
      inflating: ./clothes_dataset/green_shoes/abe553c85fda3c11999b029fd26907c44618aafb.jpg  
      inflating: ./clothes_dataset/green_shoes/addb67bb720c6a232b67abf94b084e9893ac9509.jpg  
      inflating: ./clothes_dataset/green_shoes/af91edf61e78f9fa12662dcc3b2bec0fc721595a.jpg  
      inflating: ./clothes_dataset/green_shoes/afca5f93b083726f9c7d6aec54037130598373fa.jpg  
      inflating: ./clothes_dataset/green_shoes/b0d01536d4f0dba68aaafbddadcbbe437fc75873.jpg  
      inflating: ./clothes_dataset/green_shoes/b1b7785b911006d7e182f1c1c0e1f56e275d0547.jpg  
      inflating: ./clothes_dataset/green_shoes/b1bda275f98a43076300d7bc05f81a3a394bcf5c.jpg  
      inflating: ./clothes_dataset/green_shoes/b27fcdbf13a38348a22aff3106c950aef5459e39.jpg  
      inflating: ./clothes_dataset/green_shoes/b2aae660d92548191619674a79d7f67f9b70fa9e.jpg  
      inflating: ./clothes_dataset/green_shoes/b2ee4167f30ff897c91bdf216d250df0e95a43f7.jpg  
      inflating: ./clothes_dataset/green_shoes/b3251c7b8c35e56ddf2c502762f148d808d537c4.jpg  
      inflating: ./clothes_dataset/green_shoes/b4d4f44666f0a73300b5559e3730b7215b412939.jpg  
      inflating: ./clothes_dataset/green_shoes/b5349f1dd7bfda123dc90e2c3ffa60ab473087aa.jpg  
      inflating: ./clothes_dataset/green_shoes/b57a1882cb079843fe07b31e6c24ab1eed9eb88e.jpg  
      inflating: ./clothes_dataset/green_shoes/b57d1bec87d11b9420d862867b5882bc27430aa4.jpg  
      inflating: ./clothes_dataset/green_shoes/b5b2e369d714eaccd27b88d05db4e84143cb371a.jpg  
      inflating: ./clothes_dataset/green_shoes/b644bf1f2b0d7cb8af24ec8aa9a05b2e41416245.jpg  
      inflating: ./clothes_dataset/green_shoes/b67d9e5c3ec9bef1339d5b501a432d9c1a84b630.jpg  
      inflating: ./clothes_dataset/green_shoes/b6ddf760c3366f704c2e975551fe19f55220d3da.jpg  
      inflating: ./clothes_dataset/green_shoes/b844714fb19a8138070fa5a087ade51740777d67.jpg  
      inflating: ./clothes_dataset/green_shoes/b8d666d9427e870015a5cee9ad6c164b1983fdd7.jpg  
      inflating: ./clothes_dataset/green_shoes/b8f67ff16c590beb4ac076a32b0d4a30746aa85a.jpg  
      inflating: ./clothes_dataset/green_shoes/b9cb1c77e4959e8d21c4238e887b0c897841f638.jpg  
      inflating: ./clothes_dataset/green_shoes/b9d6d6a02c86abed69b6c324f225a67967d2a8c0.jpg  
      inflating: ./clothes_dataset/green_shoes/b9f074b8580961cdbf68b4394bae7aced204bb8e.jpg  
      inflating: ./clothes_dataset/green_shoes/ba78149b6a0ae691426024ff579803859a8d4809.jpg  
      inflating: ./clothes_dataset/green_shoes/bae312fe22b246b94d1da75121f20e498ec90634.jpg  
      inflating: ./clothes_dataset/green_shoes/bb3452ca819c7af29ca484e23b1f736905385e3d.jpg  
      inflating: ./clothes_dataset/green_shoes/bb5382f18bcf97d2a534f92f2c02bccaf3002b89.jpg  
      inflating: ./clothes_dataset/green_shoes/bc1bf9bd06f1bca43129c6d9342f3f92db212192.jpg  
      inflating: ./clothes_dataset/green_shoes/bd117509be7b66993a877a08ec2a30d9285f9046.jpg  
      inflating: ./clothes_dataset/green_shoes/bd582d7f3a57dbc1c1469efa8fbd7c54ebe21fc0.jpg  
      inflating: ./clothes_dataset/green_shoes/be7b72f035eb10aa9bef57d10ec646976f331d50.jpg  
      inflating: ./clothes_dataset/green_shoes/bea2270fb46865fda52cbff1911ff7dd14ae5d4f.jpg  
      inflating: ./clothes_dataset/green_shoes/becba9dc4b37f0379bade3502ea7be8c1470c80b.jpg  
      inflating: ./clothes_dataset/green_shoes/bee0ad031a43dcbaea97384ff43494a01c3ada41.jpg  
      inflating: ./clothes_dataset/green_shoes/bf985f2b7aee053b32d93010432e3bff44c826aa.jpg  
      inflating: ./clothes_dataset/green_shoes/bfadc76d4af398a779042b0cb05b4e83fdad7a2f.jpg  
      inflating: ./clothes_dataset/green_shoes/c04c4f6aee579a4a89efaa103d71df36bce97ebb.jpg  
      inflating: ./clothes_dataset/green_shoes/c079cf9b1bd99ae07c08a385a1e34310ceed6a82.jpg  
      inflating: ./clothes_dataset/green_shoes/c090d15682122c5b44ff30dc512c32e01d9fa4af.jpg  
      inflating: ./clothes_dataset/green_shoes/c091886682f27649eb9b19ba939f1b8f8d690194.jpg  
      inflating: ./clothes_dataset/green_shoes/c21695979af26f2b9fddd11ec1fd63223877ca7f.jpg  
      inflating: ./clothes_dataset/green_shoes/c24351a91ad2e7f3fadeaa821487138b169d81c5.jpg  
      inflating: ./clothes_dataset/green_shoes/c258a484a690d8acf3a1f303a2c6b9db00a7e347.jpg  
      inflating: ./clothes_dataset/green_shoes/c2a9470d0f753ba65a4a5d0b263965122ade0974.jpg  
      inflating: ./clothes_dataset/green_shoes/c2cc402b12362e6d3e75b7ab693b19e0d340d5dc.jpg  
      inflating: ./clothes_dataset/green_shoes/c5e6ed2dd8a962bde36264d642b587d66b18c060.jpg  
      inflating: ./clothes_dataset/green_shoes/c8875dc28a90aa7228f6dce9f56e71639dbf4994.jpg  
      inflating: ./clothes_dataset/green_shoes/c895cc8af4799223d1794d504a516312d509dbe4.jpg  
      inflating: ./clothes_dataset/green_shoes/cb497ecdba079846b05af3f22c66c758052dc04a.jpg  
      inflating: ./clothes_dataset/green_shoes/cb87d1b6a30d764c259a19758f200cabb9b0531d.jpg  
      inflating: ./clothes_dataset/green_shoes/ce060ed0a6343d71d2e59d537545dbaa66f1a6b4.jpg  
      inflating: ./clothes_dataset/green_shoes/ce4aca01833953c1e2da6e46d6a4e8b192b74196.jpg  
      inflating: ./clothes_dataset/green_shoes/cebd4398ae8c75141e2c7236d3d0184acf34f48e.jpg  
      inflating: ./clothes_dataset/green_shoes/d00cc47d05c6cde7b0ee0750a2568f262ebb92b2.jpg  
      inflating: ./clothes_dataset/green_shoes/d0886ca2190567538de17fba1a4061037d3a5d1a.jpg  
      inflating: ./clothes_dataset/green_shoes/d120f0a19dec562dce904bc5b9b32fab368c974a.jpg  
      inflating: ./clothes_dataset/green_shoes/d2c27ef8cd5cbeb3987aabdeb9f549bc5cb53c4c.jpg  
      inflating: ./clothes_dataset/green_shoes/d2df40b89144edf50fefab998bd89cba197559ee.jpg  
      inflating: ./clothes_dataset/green_shoes/d30a66e23250aeed6b8b942fed9a2a5475855723.jpg  
      inflating: ./clothes_dataset/green_shoes/d3246a3c1ad4bf2ad46c10fff22963899ce88cfa.jpg  
      inflating: ./clothes_dataset/green_shoes/d35ce3ef8465412ad2c1ff97bab8a8168b4bd35c.jpg  
      inflating: ./clothes_dataset/green_shoes/d49622b676cc15789cd93b5aaae9fb85e655754a.jpg  
      inflating: ./clothes_dataset/green_shoes/d5380237df122f354a275c8cdd9035f4d200ccaf.jpg  
      inflating: ./clothes_dataset/green_shoes/d62ea9239c6a7f5c07bebf486f758a2a0ed89140.jpg  
      inflating: ./clothes_dataset/green_shoes/d7742bb6fecb16bf669ee40019b294870dc3e777.jpg  
      inflating: ./clothes_dataset/green_shoes/d8cc93adf41ceac194830f3095d898b5e960b9cf.jpg  
      inflating: ./clothes_dataset/green_shoes/da584cc53a9a2b5c49986ee05c9a2d877f07cc29.jpg  
      inflating: ./clothes_dataset/green_shoes/dacf4bce61114f9de7a6c006096ada573253dcbd.jpg  
      inflating: ./clothes_dataset/green_shoes/dad40bf43dc1170d0a106245d3cf7914524b1cd6.jpg  
      inflating: ./clothes_dataset/green_shoes/dafd9761290862b46adb6707a027e4ecdf137067.jpg  
      inflating: ./clothes_dataset/green_shoes/dc1fb6f37518aa91653e93517c6c9692f9608bdb.jpg  
      inflating: ./clothes_dataset/green_shoes/dd117d600374291df695143021e8424d48dc0849.jpg  
      inflating: ./clothes_dataset/green_shoes/de2bbcd2bc7a346609a5e0e006cf32cd37fbe9aa.jpg  
      inflating: ./clothes_dataset/green_shoes/de5f9aeb85adef709ecfe9700b87fe002c6e45a2.jpg  
      inflating: ./clothes_dataset/green_shoes/de6e1603095c5640df21b832f5e14b2c8959cd3c.jpg  
      inflating: ./clothes_dataset/green_shoes/de8b791b245b7f66ad71d0c6abf5e3c03bb26b4c.jpg  
      inflating: ./clothes_dataset/green_shoes/dec30998bc4d8073d25636bb72589c58a7d24904.jpg  
      inflating: ./clothes_dataset/green_shoes/df3d4c8a955e14beee6d8b929027ebeacc7ffcf6.jpg  
      inflating: ./clothes_dataset/green_shoes/e053fec8363ed5ca9add4eb4ddd036ca7ed1640b.jpg  
      inflating: ./clothes_dataset/green_shoes/e1bc71c932fbd998b3ea6350908eae85daed6864.jpg  
      inflating: ./clothes_dataset/green_shoes/e23431b35e0fcde1c5f860908476c3f93b8c0e1d.jpg  
      inflating: ./clothes_dataset/green_shoes/e30cef2fcd914f073602a99850b8a0e6bfa7e8b4.jpg  
      inflating: ./clothes_dataset/green_shoes/e318eaf3a701feb736db03a2c84135d70b616f31.jpg  
      inflating: ./clothes_dataset/green_shoes/e36ed88a3b14ebfa0e91d9301e696af5cd7fd10e.jpg  
      inflating: ./clothes_dataset/green_shoes/e3e118d0f9bbec4f4677b895edaf1891a85b9066.jpg  
      inflating: ./clothes_dataset/green_shoes/e4345e24ea253a66907b3e48d890e9a00e58dc05.jpg  
      inflating: ./clothes_dataset/green_shoes/e4ac506959c96bb910da2a11e92147eca1fc6a2e.jpg  
      inflating: ./clothes_dataset/green_shoes/e58a3732999b774bb1352e828a458193c41ff66b.jpg  
      inflating: ./clothes_dataset/green_shoes/e77308684e19a55cc90693e1b75e4704126b51bb.jpg  
      inflating: ./clothes_dataset/green_shoes/e7a32f19d091d5c24f2b8468cbbf8e08821cabe0.jpg  
      inflating: ./clothes_dataset/green_shoes/e7d643c612b29aa4c63644db65b290be24f76dc6.jpg  
      inflating: ./clothes_dataset/green_shoes/e8363f06d7459b381af9e23fa60ade7244cb9f7f.jpg  
      inflating: ./clothes_dataset/green_shoes/e86c00f07068e0b598cc329350fc8400e3c62ce5.jpg  
      inflating: ./clothes_dataset/green_shoes/e8aee45cb12a7a06f3ed3c842a2e2d9552f1b14f.jpg  
      inflating: ./clothes_dataset/green_shoes/eba19fade66a545ef504b75b4e46d5ae27434811.jpg  
      inflating: ./clothes_dataset/green_shoes/ebb83b22bc16bc8983719f71fed7a0852f8c5824.jpg  
      inflating: ./clothes_dataset/green_shoes/ebc498125463e61c57964f136d0fc20aa5d38784.jpg  
      inflating: ./clothes_dataset/green_shoes/ec066c654e8e2d130686d6314262d87a9ac53bf7.jpg  
      inflating: ./clothes_dataset/green_shoes/ec58d291c10df5f7c0b326d98b224844b76a0f5f.jpg  
      inflating: ./clothes_dataset/green_shoes/ee9658aca6c488ac974fc2058260122eaa559518.jpg  
      inflating: ./clothes_dataset/green_shoes/efceffc4612898da6c868ea4bce5d78194bea562.jpg  
      inflating: ./clothes_dataset/green_shoes/efd4f11628f43075bde2fb6e307d738d5a867274.jpg  
      inflating: ./clothes_dataset/green_shoes/f0059516243d976c694a733a775ad33c81294bad.jpg  
      inflating: ./clothes_dataset/green_shoes/f05524ebb9d2a345e61ad4ae3e6f9086092faafa.jpg  
      inflating: ./clothes_dataset/green_shoes/f10b96035b26e77cfd05da34b0ee0ae3ea0a25a9.jpg  
      inflating: ./clothes_dataset/green_shoes/f1f33bed259f4b38cdf1fd56b4b83fc20fecb4cc.jpg  
      inflating: ./clothes_dataset/green_shoes/f282d7aa51b94d6deee824d54e1fdf052eed8ba9.jpg  
      inflating: ./clothes_dataset/green_shoes/f3232a59b2a131fcae090f96d3f119e6f7dc28ad.jpg  
      inflating: ./clothes_dataset/green_shoes/f3f27c3342d26e364d94ce245c0ff82456946c61.jpg  
      inflating: ./clothes_dataset/green_shoes/f41f2d013ced520d943182923b3059cc31c8742c.jpg  
      inflating: ./clothes_dataset/green_shoes/f49a0d07e6bd7e3886478d6449bf1ce692a37a47.jpg  
      inflating: ./clothes_dataset/green_shoes/f5604f779f51b4f7b1ad3a3c7905edcf73e28764.jpg  
      inflating: ./clothes_dataset/green_shoes/f5ad84e9f55a23c928f8c5d8dae23b1b457f6034.jpg  
      inflating: ./clothes_dataset/green_shoes/f5ba165aae1b542a327c4a8c76e2f303c4efb64d.jpg  
      inflating: ./clothes_dataset/green_shoes/f5e67a7d88174f0535889c1bbf1695614740bc90.jpg  
      inflating: ./clothes_dataset/green_shoes/f5e9e4fd6771dbd6c056ca69768c4121235dad01.jpg  
      inflating: ./clothes_dataset/green_shoes/f73fe5702e41f60b2f4d5267dbbea1c42b310938.jpg  
      inflating: ./clothes_dataset/green_shoes/f79620b86eec660e9f70aa15ee39c70fb6707492.jpg  
      inflating: ./clothes_dataset/green_shoes/f7e27aef4519cc8210694870fd4cb3f356ca6b84.jpg  
      inflating: ./clothes_dataset/green_shoes/f7f69b460d7729acb550c45549b2a513088f85b2.jpg  
      inflating: ./clothes_dataset/green_shoes/f7fc224e26936d8483da0d46c4c320b00ce9e28f.jpg  
      inflating: ./clothes_dataset/green_shoes/f839fbdf8f7585339d5b25f01df9cead7c3c03a9.jpg  
      inflating: ./clothes_dataset/green_shoes/f8ce679f586a9c57ce7d628b8bf28a7d9dbe5e32.jpg  
      inflating: ./clothes_dataset/green_shoes/f9226de21e084fa7e2360bfa24252bdb63ddb73e.jpg  
      inflating: ./clothes_dataset/green_shoes/fa1808695d72f0aaec21619b78ecb93751b4e3e3.jpg  
      inflating: ./clothes_dataset/green_shoes/fa556932c97c8013f7a83b7abf80aced5352dffa.jpg  
      inflating: ./clothes_dataset/green_shoes/fa5aac7614ba2e767e65734e40a894c65f48e92f.jpg  
      inflating: ./clothes_dataset/green_shoes/fb00986e3f32cb35c884643c8090e6c2a57ccf6b.jpg  
      inflating: ./clothes_dataset/green_shoes/fb214eb84d361b36760fde73f2aa989d24ad2ca5.jpg  
      inflating: ./clothes_dataset/green_shoes/fb6cca39636472c776bd9bca8af108f417088094.jpg  
      inflating: ./clothes_dataset/green_shoes/fbfa6ac16e9fddbe2a35fabeb19b106eae68c7fa.jpg  
      inflating: ./clothes_dataset/green_shoes/fc7e61a49146d7fa16fa41d507ce5557a5f48c90.jpg  
      inflating: ./clothes_dataset/green_shoes/fcd55c310406a1618eb780d17c1db708eed37611.jpg  
      inflating: ./clothes_dataset/green_shoes/fd23116113adf4dfd2a06254c2a7eb962a6d096b.jpg  
      inflating: ./clothes_dataset/green_shoes/fd4c4a7cfab6f04ad5ecdc13abb27459d7c34a68.jpg  
      inflating: ./clothes_dataset/green_shoes/fd4d68f4c6e80ef2c9a1d79f13defb6e9e2aa768.jpg  
      inflating: ./clothes_dataset/green_shoes/fdcb824ba95d69bf3ff7bdf5c6f4941d5dcc04bd.jpg  
      inflating: ./clothes_dataset/green_shoes/fe65a5165449bfee536bc0d0ed49be5f7f4a96d8.jpg  
      inflating: ./clothes_dataset/green_shoes/fe87428a8c70b580a0ea3f2135411e38be7f4c66.jpg  
      inflating: ./clothes_dataset/green_shoes/fec5f4c955df4910e334de9bbec1be4ca371b890.jpg  
      inflating: ./clothes_dataset/green_shoes/ff419862fa6a63d309a1acdabd43977fa06df0cf.jpg  
      inflating: ./clothes_dataset/green_shoes/ff4301ac97f9d2e1c96670921899da460ce7b779.jpg  
      inflating: ./clothes_dataset/green_shorts/0046f59aa6aa0012dea04d6500234ca0e290eeec.jpg  
      inflating: ./clothes_dataset/green_shorts/009e07e80def9aa5afd1e24cdc40db7bb984d1ec.jpg  
      inflating: ./clothes_dataset/green_shorts/03e41a6ea12ac7d4d9be125920e85b4f86a034a8.jpg  
      inflating: ./clothes_dataset/green_shorts/04a59f15ccc6115de2514245cc132177e083cf57.jpg  
      inflating: ./clothes_dataset/green_shorts/04d9dc5b66e7b276146538af415cd52cb53d7cbd.jpg  
      inflating: ./clothes_dataset/green_shorts/062a74db6a9fb43fbf2d1e1bdb5d47819146a6bf.jpg  
      inflating: ./clothes_dataset/green_shorts/0a226047d7ba4c69c29d4d08a1934cd8166b5241.jpg  
      inflating: ./clothes_dataset/green_shorts/0bec714b7eb9d03d931ae4a3fdee1913dda2bf10.jpg  
      inflating: ./clothes_dataset/green_shorts/0fc930109a68cb6d5d1ca874d82ea532d429f4c9.jpg  
      inflating: ./clothes_dataset/green_shorts/142616368b431a80fce361f1d62960de79b702b2.jpg  
      inflating: ./clothes_dataset/green_shorts/16920fff62d89c3e78c103e750b812bc8c089608.jpg  
      inflating: ./clothes_dataset/green_shorts/17cbc6a0987da814d1b2dcb4d0ca0aef0748f940.jpg  
      inflating: ./clothes_dataset/green_shorts/19fa73c18e9dc7d17d7feb3d89d57a721f9cb393.jpg  
      inflating: ./clothes_dataset/green_shorts/1b34871685924aadf9fbd85d92f8e102380826c8.jpg  
      inflating: ./clothes_dataset/green_shorts/1baa2a0c50e4fa9d29d0ca22494b1570a9c2cef0.jpg  
      inflating: ./clothes_dataset/green_shorts/1bd89738be44c8ae0446d2cc42c83e993963748b.jpg  
      inflating: ./clothes_dataset/green_shorts/1cf0de538a32958254af6debed4f316e047d0fb6.jpg  
      inflating: ./clothes_dataset/green_shorts/1cff95154fbd998b3b708c9b01c14ff7921b6721.jpg  
      inflating: ./clothes_dataset/green_shorts/1d8dfe18217cc5a05b97c63a5cabb63078049a0e.jpg  
      inflating: ./clothes_dataset/green_shorts/1d97eeb8cc03578998849c64c21503c77f7207f4.jpg  
      inflating: ./clothes_dataset/green_shorts/1f1897cac5abe191c4afa3fb9ab5883602a7b93b.jpg  
      inflating: ./clothes_dataset/green_shorts/1f92a611866af3496de242fccebb922c1adba862.jpg  
      inflating: ./clothes_dataset/green_shorts/2094b82b5bc79f776939d3de871b4258ccc75437.jpg  
      inflating: ./clothes_dataset/green_shorts/20a7f2f964a86326b84d2d391e9c6ed32c6fddf3.jpg  
      inflating: ./clothes_dataset/green_shorts/2148edf9b25a07e209cc968465d0013ddc1a5a11.jpg  
      inflating: ./clothes_dataset/green_shorts/21598eb173a164507661f900721028ee61bce89e.jpg  
      inflating: ./clothes_dataset/green_shorts/23d0da36bf371d00beea7484b3603e88663492f3.jpg  
      inflating: ./clothes_dataset/green_shorts/281f249896a84a853cb21873de03925f3d859016.jpg  
      inflating: ./clothes_dataset/green_shorts/2830456839ca06c6a044dd337f425f55dfed4835.jpg  
      inflating: ./clothes_dataset/green_shorts/2891411dfb560199e33f81f6fb9fec4fb7e10dcd.jpg  
      inflating: ./clothes_dataset/green_shorts/28ab04a4e4af7ce0040e0de81ce98255378be1fd.jpg  
      inflating: ./clothes_dataset/green_shorts/2a088f7cfa93f18fbb0d29f4f2a6ffb1615a9780.jpg  
      inflating: ./clothes_dataset/green_shorts/2eea48bda08dde3352f2d2a3fb4d665c4073a635.jpg  
      inflating: ./clothes_dataset/green_shorts/30ac1685ba9f6a01499ccce66c267d7f34f115e2.jpg  
      inflating: ./clothes_dataset/green_shorts/30f81b2e6efad39872dbe04a6f5f3b1a35624d51.jpg  
      inflating: ./clothes_dataset/green_shorts/331ffd9e2e0dadbfc7339c64b238a1f4fafcd0c3.jpg  
      inflating: ./clothes_dataset/green_shorts/334fcb0e9837379441646411debd7658fb2efbba.jpg  
      inflating: ./clothes_dataset/green_shorts/36c6acf870e8c8b7f375fa765114ad8b37893506.jpg  
      inflating: ./clothes_dataset/green_shorts/39f7860ba7876c3c9a8c1265afbb6495057478aa.jpg  
      inflating: ./clothes_dataset/green_shorts/3cde92e0eb67815a192a5f837c9694669b35901c.jpg  
      inflating: ./clothes_dataset/green_shorts/3db064bc140598f16deb70e8b2a348a89ca0edae.jpg  
      inflating: ./clothes_dataset/green_shorts/3dbe70aacd0b88c9f4f6dd337b237802b5b8b02f.jpg  
      inflating: ./clothes_dataset/green_shorts/3dc7c361a14e629207b5df97db76411bfdebe0d7.jpg  
      inflating: ./clothes_dataset/green_shorts/43c5750606b439cb0eeeb3aaa11a445f0677da65.jpg  
      inflating: ./clothes_dataset/green_shorts/49d5751f70918a6c5031df93b4d8a797207315cc.jpg  
      inflating: ./clothes_dataset/green_shorts/4ab4a20ae9dd7e5109dbf0026a8c1fc439bdc1ac.jpg  
      inflating: ./clothes_dataset/green_shorts/4bd45f753d6b1174973a3a660fd9a1154a27b2a8.jpg  
      inflating: ./clothes_dataset/green_shorts/4eb576d355a4336f585586c68706906999e66937.jpg  
      inflating: ./clothes_dataset/green_shorts/4f9a1dca69bd8f661c2b478b988ae18fdb742170.jpg  
      inflating: ./clothes_dataset/green_shorts/5307bc6bc57060e03d506b0160ba617471bf73a9.jpg  
      inflating: ./clothes_dataset/green_shorts/597e796cc67fa7ce549b37bff008c16ad5fe1986.jpg  
      inflating: ./clothes_dataset/green_shorts/5d2ad7b16f148a6d6424b43120abb18649858131.jpg  
      inflating: ./clothes_dataset/green_shorts/5efd179be43504a7bfa30d648cbc02d86656996c.jpg  
      inflating: ./clothes_dataset/green_shorts/5fd6113057eda584fdd00af727b0658a7ffa1d23.jpg  
      inflating: ./clothes_dataset/green_shorts/5fe9a90f3d36fd9d94057721f3352b71033f6cfc.jpg  
      inflating: ./clothes_dataset/green_shorts/60b9174c6d19bad8de15fca4443b0be3a3e1ddad.jpg  
      inflating: ./clothes_dataset/green_shorts/61ab5eb74f8e3c3731ae17d12c1fdf79c05d19b9.jpg  
      inflating: ./clothes_dataset/green_shorts/646f1540b7426a82fcb0629f7c55ae062eaf0742.jpg  
      inflating: ./clothes_dataset/green_shorts/65d6be800a246d6ba7cce9f9e3571d04ee397792.jpg  
      inflating: ./clothes_dataset/green_shorts/661933466b9c3ba3f295155f6e67088b129033fe.jpg  
      inflating: ./clothes_dataset/green_shorts/6889ee3cc8d2d5fd683ad59f074b5751d8050727.jpg  
      inflating: ./clothes_dataset/green_shorts/6bb75d9a136d93bdb57a4a36ff4ca008cd81815c.jpg  
      inflating: ./clothes_dataset/green_shorts/6f728c062ef547b572aeb75c2d2b4aa596ebd02a.jpg  
      inflating: ./clothes_dataset/green_shorts/6f942a1e862543e8ddec60fdccbeac77919abc75.jpg  
      inflating: ./clothes_dataset/green_shorts/7079c34abd5652d10ca8bf83c5611dfe89ad7dd0.jpg  
      inflating: ./clothes_dataset/green_shorts/71f40f3dc65292e7584a07e9d55bab37c42f4680.jpg  
      inflating: ./clothes_dataset/green_shorts/73ce29bf11e738c427a20b0aad1476fc30d8d6df.jpg  
      inflating: ./clothes_dataset/green_shorts/743b44771120d9a1a7771514f0435b51e14693c9.jpg  
      inflating: ./clothes_dataset/green_shorts/78396643710b5db2117c3ee76e06e6d780af05f8.jpg  
      inflating: ./clothes_dataset/green_shorts/7d80687e04958dbaf113862da08329e5b90f26a7.jpg  
      inflating: ./clothes_dataset/green_shorts/7ebd2ba4aa7d9b527cbc6b2ca956d2af87ec3f6c.jpg  
      inflating: ./clothes_dataset/green_shorts/82a12ffc8730fd69fcfe80ca14490cb3c11b6616.jpg  
      inflating: ./clothes_dataset/green_shorts/85203e54b2e508d98398d29b4219814f28d56bc4.jpg  
      inflating: ./clothes_dataset/green_shorts/88f0b0bb02ded7a441600a834ca4ff3fbba479e7.jpg  
      inflating: ./clothes_dataset/green_shorts/8a5b219931e86fb45343e0344ad9310841521a1e.jpg  
      inflating: ./clothes_dataset/green_shorts/8ad97b452c511a31b6574dae63e662933f43050f.jpg  
      inflating: ./clothes_dataset/green_shorts/93f38d4fe43d938093b7a43b29a1453e083fab03.jpg  
      inflating: ./clothes_dataset/green_shorts/961ff3c6ef3aed78d9e9e54d0298ddf7bbe4ecd3.jpg  
      inflating: ./clothes_dataset/green_shorts/98ae5d29e28032f84bb3df0077625f316cfe6e6c.jpg  
      inflating: ./clothes_dataset/green_shorts/9952a832075926393b0900d53f105b2e947bff12.jpg  
      inflating: ./clothes_dataset/green_shorts/9a17afe2094aac9de8fac2aaca00aca0436fd054.jpg  
      inflating: ./clothes_dataset/green_shorts/9b8c06d601947725bc6d6cb628d309a5639f4879.jpg  
      inflating: ./clothes_dataset/green_shorts/9bbcc7c2697e825c46d04b2ec906525d81e2f778.jpg  
      inflating: ./clothes_dataset/green_shorts/a053f40c04fe426c132baeb6a601929e8572ead3.jpg  
      inflating: ./clothes_dataset/green_shorts/a2c3f62dc44b77dcb3af806968efe79b1fa8f528.jpg  
      inflating: ./clothes_dataset/green_shorts/a3236d24244bd07fe06a4ef11e4adc8652011bb1.jpg  
      inflating: ./clothes_dataset/green_shorts/a32a6dc763f0ef484c1d7fbbb6b388236e5adff2.jpg  
      inflating: ./clothes_dataset/green_shorts/a61abd066e11e3e53dbd39495b1846b60f70eb1c.jpg  
      inflating: ./clothes_dataset/green_shorts/a6a6ee8f112717e2f3a0276aacd72a0567018a42.jpg  
      inflating: ./clothes_dataset/green_shorts/a7df95ea2b81605b8eebd63dafec75c9e5dd4c3e.jpg  
      inflating: ./clothes_dataset/green_shorts/a95dd4040a741620e63484247bf128ba78be71fe.jpg  
      inflating: ./clothes_dataset/green_shorts/aa13c3c47847ae5e2625aa31b37ecd3721e57a3e.jpg  
      inflating: ./clothes_dataset/green_shorts/accfa46ec100020cf30f752646882bf55f485093.jpg  
      inflating: ./clothes_dataset/green_shorts/ad5f476d7dfc7eaf5a799a6ed95c56200baf868d.jpg  
      inflating: ./clothes_dataset/green_shorts/add144851a740c64fbcbcd2e0cef2c178cd667e9.jpg  
      inflating: ./clothes_dataset/green_shorts/adf99580ae1099fd1c3860ee61bc6a64961bd487.jpg  
      inflating: ./clothes_dataset/green_shorts/ae3d95f9d666a27650b716de74b7a06cf0fe8452.jpg  
      inflating: ./clothes_dataset/green_shorts/ae4ebb7935eb9b433a3c9cb39ecc229a8c7b954d.jpg  
      inflating: ./clothes_dataset/green_shorts/aebbc42aea798c2051bbb6b5934e90a61311b8de.jpg  
      inflating: ./clothes_dataset/green_shorts/af55f5f9b439872374ac07023acbc8174195233c.jpg  
      inflating: ./clothes_dataset/green_shorts/af6911f2fceb5e827f7ae2bfee065a2c688ced09.jpg  
      inflating: ./clothes_dataset/green_shorts/b5052a8c18c509c2170ea4c8002ffabff2e0cd28.jpg  
      inflating: ./clothes_dataset/green_shorts/b67856bd68ad278ffd19faf8e9544c73f0628e82.jpg  
      inflating: ./clothes_dataset/green_shorts/bc478040b34784f9ee12339988fd31098622e45c.jpg  
      inflating: ./clothes_dataset/green_shorts/bd671d41f17389a5e04afbb41ab7ebd7c6ff3742.jpg  
      inflating: ./clothes_dataset/green_shorts/c378ba55a519d8095e7346526ffd1178e8c6a946.jpg  
      inflating: ./clothes_dataset/green_shorts/c6771dbda5a1796292428e3b819008207916c863.jpg  
      inflating: ./clothes_dataset/green_shorts/c6e83376906bbc1ce2d963ebcbd8e286180b754c.jpg  
      inflating: ./clothes_dataset/green_shorts/c70b4cd11c1303e83451863e3a2d4c13ed2ece05.jpg  
      inflating: ./clothes_dataset/green_shorts/c7d7ff0ad610a9a973761ab83ff40369ffebde47.jpg  
      inflating: ./clothes_dataset/green_shorts/c9302d7b14f1e25ee9b6d57fd7a3432f143c6746.jpg  
      inflating: ./clothes_dataset/green_shorts/c951172a75cef60981fb577887367fbba868c3dd.jpg  
      inflating: ./clothes_dataset/green_shorts/ca8665169a00ded80d7e3f73fd4f10ed6bddff67.jpg  
      inflating: ./clothes_dataset/green_shorts/cae1e81e2048bc9e5a6bf5f0fc137da8f1c9e252.jpg  
      inflating: ./clothes_dataset/green_shorts/d3c1df2a88ae220d9f5e248e1010d191a7747e97.jpg  
      inflating: ./clothes_dataset/green_shorts/d5d3cf09622576aee0d751f996e0934290f17848.jpg  
      inflating: ./clothes_dataset/green_shorts/dbc84b190251638a5246a1a18d6ea805151de8de.jpg  
      inflating: ./clothes_dataset/green_shorts/ddc3621312266ccf3742a3637d915524d0a0bf7d.jpg  
      inflating: ./clothes_dataset/green_shorts/de59a6f54a3042ee9b667350a7dfe96078806800.jpg  
      inflating: ./clothes_dataset/green_shorts/e22240023222833be0cd3bc9d51a330f1fa517ff.jpg  
      inflating: ./clothes_dataset/green_shorts/e24b291aad66cff786f911689b329af33ca34adf.jpg  
      inflating: ./clothes_dataset/green_shorts/e6f14c77ee414ac1e985b8a124170bd5525ef290.jpg  
      inflating: ./clothes_dataset/green_shorts/e74d11d3772c1899e9940da08517ece1c2ad5c39.jpg  
      inflating: ./clothes_dataset/green_shorts/eb6dca2dffbfcfd3e98f82814744b21f75b3c712.jpg  
      inflating: ./clothes_dataset/green_shorts/ec8afefd605d57ef426d1f52b6c22c915966c185.jpg  
      inflating: ./clothes_dataset/green_shorts/ee2b2e83e7e70029f6b29ffea092c45a6a833d0c.jpg  
      inflating: ./clothes_dataset/green_shorts/ef21b9d0f27e7da9f76b6605b8bed06d5472a2eb.jpg  
      inflating: ./clothes_dataset/green_shorts/f251850c2d70bc8849f267305084119753b2ef29.jpg  
      inflating: ./clothes_dataset/green_shorts/f264cc04bffccf893fbef357ac344a24bdbec63c.jpg  
      inflating: ./clothes_dataset/green_shorts/f2aa6b2887ccb406e1bb9582af0074db610f759c.jpg  
      inflating: ./clothes_dataset/green_shorts/f338b50053cbb5b9acafffac4b27b7dc0169933f.jpg  
      inflating: ./clothes_dataset/green_shorts/f4b880f50c0dac817b627ce841310edb5b9f0247.jpg  
      inflating: ./clothes_dataset/green_shorts/f639966b56bf8c1d28bcd7fc68f3d114889224f9.jpg  
      inflating: ./clothes_dataset/green_shorts/fc56336fc99d40dc82db6ca2fe28583d2b7bc269.jpg  
      inflating: ./clothes_dataset/green_shorts/ffcd9e673050122fdd38a7e8318aa5229a7ca5cc.jpg  
      inflating: ./clothes_dataset/red_dress/0017962f3bc1155269c918c639682255ecb0e103.jpg  
      inflating: ./clothes_dataset/red_dress/00366a493888f68ba46e8f91048f655f13af1d75.jpg  
      inflating: ./clothes_dataset/red_dress/003aec452c6332f11c68c18971fe0ebbb322812f.jpg  
      inflating: ./clothes_dataset/red_dress/00613a3517aaa259def4a91bb8889dfafb36226e.jpg  
      inflating: ./clothes_dataset/red_dress/01553d10275a342f17045ed21674b72f63e86bbd.jpg  
      inflating: ./clothes_dataset/red_dress/016e4896d52cd608b4cb96ffc13e1f847f7a3db7.jpg  
      inflating: ./clothes_dataset/red_dress/01a20e0371b1b5fe0f853d38ac052473a6775d69.jpg  
      inflating: ./clothes_dataset/red_dress/01b2036144322f9a3095a203e49134176b25d29a.jpg  
      inflating: ./clothes_dataset/red_dress/01ce31f297f8dad80b5ce0984d3b31f8dd40b9a3.jpg  
      inflating: ./clothes_dataset/red_dress/02028f70c10618f15f9fa50fef4199cccc4c564a.jpg  
      inflating: ./clothes_dataset/red_dress/02510dcbded24e4b72fa406f1d1537f1e288941f.jpg  
      inflating: ./clothes_dataset/red_dress/028634be3079a276147ed5b5b5552d9592469f77.jpg  
      inflating: ./clothes_dataset/red_dress/028cce11e45a3564228bd8e38d6a751497554b68.jpg  
      inflating: ./clothes_dataset/red_dress/0319661efccba971ba6299e3381cfc75812dbddf.jpg  
      inflating: ./clothes_dataset/red_dress/037c590a5e5e1dbf85f151e0626908d22a70a5b3.jpg  
      inflating: ./clothes_dataset/red_dress/040d5f5d05b6f980bc1c9ac4356370eb911f9d54.jpg  
      inflating: ./clothes_dataset/red_dress/045a35a395956a34a65ef6d3a46dc6cf1746aa60.jpg  
      inflating: ./clothes_dataset/red_dress/0486e696271277f84f8c9076fd360b46b7a77f5a.jpg  
      inflating: ./clothes_dataset/red_dress/05e0c127256047378e202a12717fa67e0ab3fcca.jpg  
      inflating: ./clothes_dataset/red_dress/0677913826da51f5efbd27b3d7824d0503051737.jpg  
      inflating: ./clothes_dataset/red_dress/06970f1a6bb3223b1195b1a04980558af651abfb.jpg  
      inflating: ./clothes_dataset/red_dress/06b345dcf5666e8a0a007f277de104877a8156a0.jpg  
      inflating: ./clothes_dataset/red_dress/08da33fafb6725f0e36d70aa6fc05891427fbddc.jpg  
      inflating: ./clothes_dataset/red_dress/08e11744cde5cecd68d7fa43168a6267156ec307.jpg  
      inflating: ./clothes_dataset/red_dress/092fd7aaf93018143ddf43b638c7167510a1e0cf.jpg  
      inflating: ./clothes_dataset/red_dress/094333d4ed2d69f1e7e102272690a2e912d736e6.jpg  
      inflating: ./clothes_dataset/red_dress/098ecbc605a0d4277661dbefa5bcec712e166d52.jpg  
      inflating: ./clothes_dataset/red_dress/09ca08a588e34ebedafe68314ddf1e0c19611b9d.jpg  
      inflating: ./clothes_dataset/red_dress/09d48b4bcc4d53fb2cf5d1848b21d9695df0c837.jpg  
      inflating: ./clothes_dataset/red_dress/0a2f48eba952bc1a3b4770f05d0b95d38595596b.jpg  
      inflating: ./clothes_dataset/red_dress/0a3c276efce82b8ac57146fb88ce20a3c836bc3e.jpg  
      inflating: ./clothes_dataset/red_dress/0a4b55c4c27f5d94a9e26d16c43f1258aade5b29.jpg  
      inflating: ./clothes_dataset/red_dress/0a5255a799db62ec5438a80793bf761aac1e4213.jpg  
      inflating: ./clothes_dataset/red_dress/0a9e5e56022fe2c7d9b62f37eb98232ccc67790d.jpg  
      inflating: ./clothes_dataset/red_dress/0ac93aad414df7dc704074305ccda4028cdcb142.jpg  
      inflating: ./clothes_dataset/red_dress/0b2ae816758b92d184b55962ca6cb35b9727053e.jpg  
      inflating: ./clothes_dataset/red_dress/0b2f782319c5ef6228081bd3bb231f3649040cb1.jpg  
      inflating: ./clothes_dataset/red_dress/0b647894d894b756379f4d5bcd0c893703879f57.jpg  
      inflating: ./clothes_dataset/red_dress/0b9d3a914bfbe357a9dc54210b03f3c6b16744a3.jpg  
      inflating: ./clothes_dataset/red_dress/0ba342960c114c6a3257e2881a4602b60d197c62.jpg  
      inflating: ./clothes_dataset/red_dress/0ba585c6ceacb858871bb19fadbf7cdd8f482391.jpg  
      inflating: ./clothes_dataset/red_dress/0bf0a2c30240871d352b8a7640f1527791a3fc91.jpg  
      inflating: ./clothes_dataset/red_dress/0c520b0df4ea17eb9efb0d4b0a82527a145d99cc.jpg  
      inflating: ./clothes_dataset/red_dress/0c866f26cdd0de57a5f97fb57dae3760d271a0ff.jpg  
      inflating: ./clothes_dataset/red_dress/0c9c96d9cc7d74d097c755269d5c8b73d22e5a09.jpg  
      inflating: ./clothes_dataset/red_dress/0cc9b90977a070b0dda5b3c63ec9e13c9eda0189.jpg  
      inflating: ./clothes_dataset/red_dress/0cf1f913447f285c02b8ec6361458ff020577d3c.jpg  
      inflating: ./clothes_dataset/red_dress/0d6b535c92b898953e819531678bfe534e29de4d.jpg  
      inflating: ./clothes_dataset/red_dress/0dddd33687c35ef942a64d9469d64b701e986322.jpg  
      inflating: ./clothes_dataset/red_dress/0e0ea1ec7712bfeeed0e783585b9c8879f9f710f.jpg  
      inflating: ./clothes_dataset/red_dress/0e1c537fbdc3c9b33897c808ebe36f8aafec3e78.jpg  
      inflating: ./clothes_dataset/red_dress/0e4861b9eace365d14cf3714d2c8ae551a39ad10.jpg  
      inflating: ./clothes_dataset/red_dress/0e8c6a038161e9b6c7570a31fe1a6fbe45038860.jpg  
      inflating: ./clothes_dataset/red_dress/0ebc94bcadfb98b4f791abdb5dcff9e256c9df17.jpg  
      inflating: ./clothes_dataset/red_dress/0ed7a39d5b075ea730ba256737dd566ae5bf3c5e.jpg  
      inflating: ./clothes_dataset/red_dress/0ef9e536be9da94f6a2be69e6ab3b6c6d58fcb22.jpg  
      inflating: ./clothes_dataset/red_dress/0f6923a6438838a3f6b824c40fce6622f93cde78.jpg  
      inflating: ./clothes_dataset/red_dress/0f69e6b1295a310e2e55659a27881685ead46c18.jpg  
      inflating: ./clothes_dataset/red_dress/0f7b9a5ee8c83e2deb00c6b4919805930a28e582.jpg  
      inflating: ./clothes_dataset/red_dress/0fd4b68d03a9a4105c30f653f0197d97cdc8f868.jpg  
      inflating: ./clothes_dataset/red_dress/0ff223667e6a578233daeec5309ced80b8aad8e0.jpg  
      inflating: ./clothes_dataset/red_dress/0ffc05609743fb8f618ff9087fe1f3430a1dc736.jpg  
      inflating: ./clothes_dataset/red_dress/10512979a1b6620a3a49a85cb4d5385fca27f46d.jpg  
      inflating: ./clothes_dataset/red_dress/1099d989b74a99a57b5489248d50d6d5b2cb6434.jpg  
      inflating: ./clothes_dataset/red_dress/1121ded2cec80962a61366e59f23170ee6121213.jpg  
      inflating: ./clothes_dataset/red_dress/119816920f7c5b14e872edda4656652abb1d2803.jpg  
      inflating: ./clothes_dataset/red_dress/121c63fea9574fe7ee9480d253473b48bf87851e.jpg  
      inflating: ./clothes_dataset/red_dress/12284ded1f9e9dadfff9cbf9301a88c1307b2659.jpg  
      inflating: ./clothes_dataset/red_dress/12b0a2b1804a8a0db7c772f9cbbb0b95007cb03d.jpg  
      inflating: ./clothes_dataset/red_dress/12eead3d2a6a8b580c2232f5107f2b2771dbbaa7.jpg  
      inflating: ./clothes_dataset/red_dress/132c6a0548981feafe10d79c22dfef9f8ef0cdcf.jpg  
      inflating: ./clothes_dataset/red_dress/13836f0a210f634783b2147e739acaebfc357e32.jpg  
      inflating: ./clothes_dataset/red_dress/13aefa2b0c952219d78eb01bf4c698ba7639ae63.jpg  
      inflating: ./clothes_dataset/red_dress/13b00b9f06b0757a955391510fa0107478c693a6.jpg  
      inflating: ./clothes_dataset/red_dress/13d0731a99280f3655df9c910cd6cdcaf725e83a.jpg  
      inflating: ./clothes_dataset/red_dress/145d5eecc4e4760b33e15ff520df7473338483ef.jpg  
      inflating: ./clothes_dataset/red_dress/14a640477291ae5791843f2ce4509562bfffb894.jpg  
      inflating: ./clothes_dataset/red_dress/14b849caef6591ddd435d4be61389ba2e931d3b8.jpg  
      inflating: ./clothes_dataset/red_dress/153a4b4120ed2ab9b4c165b902c1e4925de43b89.jpg  
      inflating: ./clothes_dataset/red_dress/1584e56b170675889c72ccfcc0b4e94efa2ff193.jpg  
      inflating: ./clothes_dataset/red_dress/15af5a8eb7d4f504d251df5ecd272a2fab443d02.jpg  
      inflating: ./clothes_dataset/red_dress/160ac8c056a0fa4702e5e008bfcc870007df564b.jpg  
      inflating: ./clothes_dataset/red_dress/16ebef644c758ec2c15e20c25f9176f5a67d422e.jpg  
      inflating: ./clothes_dataset/red_dress/1735ee51a8394fdc102a83af35ace858a55c483f.jpg  
      inflating: ./clothes_dataset/red_dress/17413bfc23f8c296e87b8472593617e5429903c0.jpg  
      inflating: ./clothes_dataset/red_dress/1744c7deb17ac9e2591afb482138d264c02aa1c5.jpg  
      inflating: ./clothes_dataset/red_dress/175153ba7c1ee652b635c7bf103848669bc8627a.jpg  
      inflating: ./clothes_dataset/red_dress/17689c7fe9e26e692c5ea8aba33104fbc350799c.jpg  
      inflating: ./clothes_dataset/red_dress/17b1af4b07feba945696cbfd06c56dc8354da70b.jpg  
      inflating: ./clothes_dataset/red_dress/17b3ce07c62e3489c86cf961a3f10e7c377bc3bd.jpg  
      inflating: ./clothes_dataset/red_dress/17be8b6093032395f6b153d86827c13011afd6ed.jpg  
      inflating: ./clothes_dataset/red_dress/18065da085c5cd332074070ee04d5586ebf2f155.jpg  
      inflating: ./clothes_dataset/red_dress/1855be4b1d28375e8f072183bedc0af42058dba6.jpg  
      inflating: ./clothes_dataset/red_dress/1894ba6dab42a38b4a1397acb6b796b8cc7e1852.jpg  
      inflating: ./clothes_dataset/red_dress/18aaacbc9b79ab9a544f9131394de6267b6ef7b4.jpg  
      inflating: ./clothes_dataset/red_dress/18fa96b30103df9cb6a4f9e9b208695732713c92.jpg  
      inflating: ./clothes_dataset/red_dress/195feae84fa4964f726c79b47f89c87aa345f11f.jpg  
      inflating: ./clothes_dataset/red_dress/198449d52c6628dc7318ae190d31c2f5d2170491.jpg  
      inflating: ./clothes_dataset/red_dress/19c11a1cf7aec1b2426dae22afd66e51fe3b8791.jpg  
      inflating: ./clothes_dataset/red_dress/19cc4a7620430fc24ec0595f46fd9be9406e4307.jpg  
      inflating: ./clothes_dataset/red_dress/19f55440c290e48223094df28150c5ff090cbc7e.jpg  
      inflating: ./clothes_dataset/red_dress/1a0dffac753d060223b702075b17415ce320c297.jpg  
      inflating: ./clothes_dataset/red_dress/1a70b09e09bba93ed0f7cca6250957ecd5ae1b9c.jpg  
      inflating: ./clothes_dataset/red_dress/1a84a76a29767c6f873446e33c468b112f949d8a.jpg  
      inflating: ./clothes_dataset/red_dress/1aabcb8e6104f65d871e1006f442566c48a24e6e.jpg  
      inflating: ./clothes_dataset/red_dress/1ac4a2eee1ac6c8fb0b7049bba2c4242cd4bd6f5.jpg  
      inflating: ./clothes_dataset/red_dress/1b8daaa375f32dc8898ed08b187c82ee209fdcbc.jpg  
      inflating: ./clothes_dataset/red_dress/1c0b0eb28abe60ef1d14538d201614496af8dcef.jpg  
      inflating: ./clothes_dataset/red_dress/1c1f671e932a5abe6d1e0d450a82b7e7465f7134.jpg  
      inflating: ./clothes_dataset/red_dress/1c5194ba77e282095d54909800b3c4ea77633cf0.jpg  
      inflating: ./clothes_dataset/red_dress/1c63f56a72493ca4ecc367338a6b46014643222a.jpg  
      inflating: ./clothes_dataset/red_dress/1c67b58280fb3308bfdda4fda7074948d5429e51.jpg  
      inflating: ./clothes_dataset/red_dress/1cd9db32b2cfcff21f491ae3db493b6cf3f0ad80.jpg  
      inflating: ./clothes_dataset/red_dress/1d1d1eb5caaae70a7beeb704134dab55e8e4df41.jpg  
      inflating: ./clothes_dataset/red_dress/1da2e75d7e4ec49aed4ed0bb57ccce6ac0b02e3a.jpg  
      inflating: ./clothes_dataset/red_dress/1e5968220eebadd8202730a092991f672e4694c9.jpg  
      inflating: ./clothes_dataset/red_dress/1eac322c42f8df968a07355af30cc8003f73657c.jpg  
      inflating: ./clothes_dataset/red_dress/1ef34fd92ae5a4875ce5d915d93b6806d1ab3c15.jpg  
      inflating: ./clothes_dataset/red_dress/1f1b6e3b1b811570756d842c6daf564c424d2eb3.jpg  
      inflating: ./clothes_dataset/red_dress/1ffeb4055f36bf9860d96d6a00b0923cb2274035.jpg  
      inflating: ./clothes_dataset/red_dress/203ad9f43a2d4132d3cf43f97732ddc9eb377fb9.jpg  
      inflating: ./clothes_dataset/red_dress/2060df8500b9edc4cd9dbd0811b25f259f1d47b6.jpg  
      inflating: ./clothes_dataset/red_dress/207e424120100f9d808051927042ee885a0b80e5.jpg  
      inflating: ./clothes_dataset/red_dress/212e7ed5e3c1e9ad5536ea71d558ee807718cffc.jpg  
      inflating: ./clothes_dataset/red_dress/218ba8e1c58be2de4380eb16bd8e481adf526a85.jpg  
      inflating: ./clothes_dataset/red_dress/21fe38f75094a9389c2c2d71ea158ad2a570a06b.jpg  
      inflating: ./clothes_dataset/red_dress/2289b86ef65548b9ecacd0f7b1f2ac228279b04d.jpg  
      inflating: ./clothes_dataset/red_dress/229ba55490a388a59d2f400b42cede4852a630cc.jpg  
      inflating: ./clothes_dataset/red_dress/22d42cae99a586ca540f80323e06b3c845ef98dd.jpg  
      inflating: ./clothes_dataset/red_dress/2343d61875bd69473387dabca986797e90a1603d.jpg  
      inflating: ./clothes_dataset/red_dress/2377a9215ab3882ccb2432f488d744bfd371ee7d.jpg  
      inflating: ./clothes_dataset/red_dress/23a088468422db9a90b7da6432532ce822f64661.jpg  
      inflating: ./clothes_dataset/red_dress/23e64af0d8a97323851c358e6ef7b0ddd92bb6c9.jpg  
      inflating: ./clothes_dataset/red_dress/23fb35b482bad148b7afeb9c64556abcd07484ad.jpg  
      inflating: ./clothes_dataset/red_dress/24da2629de29b0465b433440574706d1944b5ac8.jpg  
      inflating: ./clothes_dataset/red_dress/253353f7493c041f943f562b83bd22d60119ebd3.jpg  
      inflating: ./clothes_dataset/red_dress/257e5777414a089ea0b3f05c82ff9abc3a47dc6c.jpg  
      inflating: ./clothes_dataset/red_dress/25abb336172ccf232eb79e1697bfc6129dfb7f78.jpg  
      inflating: ./clothes_dataset/red_dress/25c4b1b6be7db8c769ba545277cbb25bca3bf7ff.jpg  
      inflating: ./clothes_dataset/red_dress/261e46a69c1cc8ce112ac78278f46a0f9e1a301c.jpg  
      inflating: ./clothes_dataset/red_dress/263b18808e033b1f0ab28aff79dce7de79ea3c08.jpg  
      inflating: ./clothes_dataset/red_dress/26c51d3d4f4b9120d37bd62298222c8229226e1b.jpg  
      inflating: ./clothes_dataset/red_dress/26cfef6847c11ef8ff4a25a513c2ff48b8650a3c.jpg  
      inflating: ./clothes_dataset/red_dress/272c94273ddc70b2bd6755418087d88e10aebb38.jpg  
      inflating: ./clothes_dataset/red_dress/27a94fc8139c49459e45ac39a42558b667e58a4b.jpg  
      inflating: ./clothes_dataset/red_dress/27b53e2e8939c0ce6a0d1e7085fb1f8845baee43.jpg  
      inflating: ./clothes_dataset/red_dress/281cd171089a00f17a8f6e18fae8ac2ac6b94172.jpg  
      inflating: ./clothes_dataset/red_dress/2853b59e185da43746923fab074800925ba5f4e4.jpg  
      inflating: ./clothes_dataset/red_dress/286e13fe901847c883c90f22dfc702363b627ddf.jpg  
      inflating: ./clothes_dataset/red_dress/28bd3c983ce1006d69d6b60ec7a4d70af6178f95.jpg  
      inflating: ./clothes_dataset/red_dress/2946846e8f2ab7566cd58e5dcb907ca27fa25b12.jpg  
      inflating: ./clothes_dataset/red_dress/29bbcf6d0b2ffb44dce27234b929fddbe9a6e672.jpg  
      inflating: ./clothes_dataset/red_dress/2a436edfaa77ee2aa1353a3fcc1e77a97e8a454a.jpg  
      inflating: ./clothes_dataset/red_dress/2a61ed44a2521ad74f013f12b1ecc1123b09303f.jpg  
      inflating: ./clothes_dataset/red_dress/2a78ba191fb49ea7c508da3ca8e310b81c9415f2.jpg  
      inflating: ./clothes_dataset/red_dress/2a8142d3f44a3cffaa2f9ada520f4d54059d11d0.jpg  
      inflating: ./clothes_dataset/red_dress/2a958bb256fc07392a89d18dc9b755dd625f09de.jpg  
      inflating: ./clothes_dataset/red_dress/2ad857470fbf511e0ab1b337f0affbb916afb1c1.jpg  
      inflating: ./clothes_dataset/red_dress/2b9e3f1c96e6b87c726a55991f9f6ab58f3107f7.jpg  
      inflating: ./clothes_dataset/red_dress/2cffc57b737b0ef46e732f9736aed961f9e07f3b.jpg  
      inflating: ./clothes_dataset/red_dress/2d8b54bf729f8bc55a4cfa15969a320d4e4d62f3.jpg  
      inflating: ./clothes_dataset/red_dress/2ddcff8d97c0d8172d3ccb8e27b96731425f65bb.jpg  
      inflating: ./clothes_dataset/red_dress/2e001c84052b7b93bd8ea1f6994b6870fa448c84.jpg  
      inflating: ./clothes_dataset/red_dress/2e0cd64e235f3341b8e5446abc88e0f832ced222.jpg  
      inflating: ./clothes_dataset/red_dress/2e264c4a6fdefc365ffe92d56540ca3bdb0f0a0b.jpg  
      inflating: ./clothes_dataset/red_dress/2ee9ea1e3b0d97ba856b987ec97a3fc4b823d9c3.jpg  
      inflating: ./clothes_dataset/red_dress/2f7a9200e728c1a990d80f86b6e93cc2810427e6.jpg  
      inflating: ./clothes_dataset/red_dress/30c715d761f26384dc8db655580f7540e6e6ceb6.jpg  
      inflating: ./clothes_dataset/red_dress/30cdfb713c8857d35ec62125292158acb39b69c4.jpg  
      inflating: ./clothes_dataset/red_dress/30e43379462586dca990d59abccc46a1067cf0f8.jpg  
      inflating: ./clothes_dataset/red_dress/323fe19608009062d68aab2508233c833e0d98a6.jpg  
      inflating: ./clothes_dataset/red_dress/3243c79e7900c183e78b9ba0f151104631e4079b.jpg  
      inflating: ./clothes_dataset/red_dress/33ec1e8ca7f97b99cb8d53070af6573ef367707f.jpg  
      inflating: ./clothes_dataset/red_dress/344e6761d4bda5eb0bd2c15d08a810d6cedae7a3.jpg  
      inflating: ./clothes_dataset/red_dress/346ecaa576ddbe88a4ef719744e968faed5c92ff.jpg  
      inflating: ./clothes_dataset/red_dress/350d0789ed3730055f6ec1adbb1c2fec15994104.jpg  
      inflating: ./clothes_dataset/red_dress/354688e017228ee26ec9ef1ba9deacd019a861f7.jpg  
      inflating: ./clothes_dataset/red_dress/35509a0b5b502fe77e1f750570fadfb9f7db9154.jpg  
      inflating: ./clothes_dataset/red_dress/366e8da41d54a613d0322f3c2966cd59bd84717a.jpg  
      inflating: ./clothes_dataset/red_dress/367d011f3b2a6bf5457ee4076a8621d13208da13.jpg  
      inflating: ./clothes_dataset/red_dress/36c3a6603c1b3948e25a5c2640932d4fdf3ed64b.jpg  
      inflating: ./clothes_dataset/red_dress/37b0a9b34adb43ab411d54a9cc75340f84a5c209.jpg  
      inflating: ./clothes_dataset/red_dress/37b9d5b4bed6fe3c5a2a2f859b5798eb3259838f.jpg  
      inflating: ./clothes_dataset/red_dress/3825aa342231249be9cb82d2145c5651bc23a901.jpg  
      inflating: ./clothes_dataset/red_dress/383c4d91ae6a58e8c7ccd3bc7d3d5d4ded5bb1f9.jpg  
      inflating: ./clothes_dataset/red_dress/387e0d943ae43620ca1f9006a2feac852bcbad6c.jpg  
      inflating: ./clothes_dataset/red_dress/38b55e3638ca73424a422a069da352ec6bdead81.jpg  
      inflating: ./clothes_dataset/red_dress/395fb7e087296d18d2866441e95ea0e21fa7893d.jpg  
      inflating: ./clothes_dataset/red_dress/39a361eb14b71920b5fd03c39bb4b7cc039f2b7b.jpg  
      inflating: ./clothes_dataset/red_dress/39a4314126fd87a4f16745cb19eca619186aaed5.jpg  
      inflating: ./clothes_dataset/red_dress/39db4c088527e4e131261ccab2a19ac99d0664f4.jpg  
      inflating: ./clothes_dataset/red_dress/3a37ea78e8eccd8caf8d794c6db6f31f2bc8458b.jpg  
      inflating: ./clothes_dataset/red_dress/3aa5e3462b12f72f9fa92febc2051f2a28722fc1.jpg  
      inflating: ./clothes_dataset/red_dress/3ae174d09c77a47f1560a446fcdd4660be6b1487.jpg  
      inflating: ./clothes_dataset/red_dress/3b5fa39feafe9452d6430dab513beb102fbdfaeb.jpg  
      inflating: ./clothes_dataset/red_dress/3bbb5882603052d3d025f174d3c14b208270bf76.jpg  
      inflating: ./clothes_dataset/red_dress/3c0b254dd4ee477086319b4a5a82bcb215a66e1f.jpg  
      inflating: ./clothes_dataset/red_dress/3c157127072ce7a06a8237e251cdb5530e21a060.jpg  
      inflating: ./clothes_dataset/red_dress/3dbf32601cefce6aae2562f67eab9af4b7a542dd.jpg  
      inflating: ./clothes_dataset/red_dress/3dcdc996a692d6dd55cf13983f0e6fbdb2516057.jpg  
      inflating: ./clothes_dataset/red_dress/3e30719ce3f00b24016184b012ead81d9dd3d9fc.jpg  
      inflating: ./clothes_dataset/red_dress/3e6d13c547d1df54abd25622fd3c84e649d6c091.jpg  
      inflating: ./clothes_dataset/red_dress/3ee97f166d117db6f8a098cb50aad36fe085cf07.jpg  
      inflating: ./clothes_dataset/red_dress/3fa03a17616530a83ab66b6e7b4a60a5f4b534aa.jpg  
      inflating: ./clothes_dataset/red_dress/4005058fd6090a4ee8eb3613f631f9f446f27d8f.jpg  
      inflating: ./clothes_dataset/red_dress/4024aa15b010935ec06964e8a77f7169d1c0dabe.jpg  
      inflating: ./clothes_dataset/red_dress/409f3144b679b8b66dd55688cefd4007db69c7c7.jpg  
      inflating: ./clothes_dataset/red_dress/40a3c63a722b83ed738628466b0b0d21b25e0f03.jpg  
      inflating: ./clothes_dataset/red_dress/410a760f47db40a43061ee10573e3b66a0be0e81.jpg  
      inflating: ./clothes_dataset/red_dress/414f95e90737620d985a68af66e7c889850f4c68.jpg  
      inflating: ./clothes_dataset/red_dress/4199c811a5e9aa2fd93e4776abd371f5f3d63f6a.jpg  
      inflating: ./clothes_dataset/red_dress/426618251392cdc2c45e13d615d401771f5c4556.jpg  
      inflating: ./clothes_dataset/red_dress/433019122ca23a59a35b83caafc703afac137cba.jpg  
      inflating: ./clothes_dataset/red_dress/4424b5c1cb60c5c4393bef6d17976046c2e16111.jpg  
      inflating: ./clothes_dataset/red_dress/443b159ad7b34268db16841b9fabc124f2b84777.jpg  
      inflating: ./clothes_dataset/red_dress/445e9d4e42b021f11280cb1616a33f852f53a86f.jpg  
      inflating: ./clothes_dataset/red_dress/45a6cbe37f111a28d286a38d515f5717d9a7e0e5.jpg  
      inflating: ./clothes_dataset/red_dress/45d64a230043ea52a1a91a800db6dcd1deae5511.jpg  
      inflating: ./clothes_dataset/red_dress/46608a04488947cba5b95ff2daa5848e7b533376.jpg  
      inflating: ./clothes_dataset/red_dress/468a79ff9bf7e9146f2261c2385d94adc205115a.jpg  
      inflating: ./clothes_dataset/red_dress/46f2a76c4bc2e43216f9371c5763506595d7948e.jpg  
      inflating: ./clothes_dataset/red_dress/46fd4268027ab338cb64a829be78f7003a8d28d5.jpg  
      inflating: ./clothes_dataset/red_dress/4752c60f2e4984ec6cdc3a42e2ff96c10bc9f4f5.jpg  
      inflating: ./clothes_dataset/red_dress/4776ff54de68919a054e33910327ffb10ff69ad6.jpg  
      inflating: ./clothes_dataset/red_dress/477ee8423236a9f4538611e5722d86c3e3de674c.jpg  
      inflating: ./clothes_dataset/red_dress/47acb775dd2ac748f83b28e4581936da5b38cffc.jpg  
      inflating: ./clothes_dataset/red_dress/47d47ea45c93930cfd62996fcbacc93952c94f0f.jpg  
      inflating: ./clothes_dataset/red_dress/47d6e30d408075226f1179d90ad4d900b7721295.jpg  
      inflating: ./clothes_dataset/red_dress/4866dc4cc694a07575cb7eb1db2bb6b4f3f5c6b0.jpg  
      inflating: ./clothes_dataset/red_dress/48b81f6181c714258ed6e5f5f081fca7967afc6f.jpg  
      inflating: ./clothes_dataset/red_dress/49d7044234d0f0be3ef6a907096fb022c7216014.jpg  
      inflating: ./clothes_dataset/red_dress/4a91c0b1b0cbe6405072bdb5f7b673aed73e1a62.jpg  
      inflating: ./clothes_dataset/red_dress/4ac02ac19b71bca55e67cd337aebd6f01caeb714.jpg  
      inflating: ./clothes_dataset/red_dress/4aeab1b4e8658490c32ea3cd0d07bc64cc2ea118.jpg  
      inflating: ./clothes_dataset/red_dress/4b0fff23befc6b94cdb0b5119887382a3594fefa.jpg  
      inflating: ./clothes_dataset/red_dress/4b43f7c9924f97af29acfc6f885960c4f547ba97.jpg  
      inflating: ./clothes_dataset/red_dress/4b5d9b8ed532be6aa415a651114f5fa26a35cd16.jpg  
      inflating: ./clothes_dataset/red_dress/4bcf2acdb3d8430e5786d7a84222ccafc94e2a35.jpg  
      inflating: ./clothes_dataset/red_dress/4be981e6a6e0b5b190538bff3ca9aa2033eeb2c0.jpg  
      inflating: ./clothes_dataset/red_dress/4c699c39927657c292cba5dfe4e6218c63eb6723.jpg  
      inflating: ./clothes_dataset/red_dress/4c8006b9d1513dc0b425e7fc3a0a19669cc395a5.jpg  
      inflating: ./clothes_dataset/red_dress/4d50114e638d52027d111e3d894c9b4c4221c174.jpg  
      inflating: ./clothes_dataset/red_dress/4d6a471bfef79fe112cb79404a247196f17c4ee9.jpg  
      inflating: ./clothes_dataset/red_dress/4dbdc9d52e990dc67520c8f2c68c4781891ae71b.jpg  
      inflating: ./clothes_dataset/red_dress/4dfda62de79d6b8f33c6ce001415f15d8176f542.jpg  
      inflating: ./clothes_dataset/red_dress/4e486121cb7bee7a774eb634d5ef27729bf5be08.jpg  
      inflating: ./clothes_dataset/red_dress/4e73af4f1e4cf2b7aab6596978e7f22c9d0716f7.jpg  
      inflating: ./clothes_dataset/red_dress/4eae11ad65b6c538375b0cb51209e4399732c960.jpg  
      inflating: ./clothes_dataset/red_dress/4ec0f8753293ee7f960111931cc6dd7dc33d4457.jpg  
      inflating: ./clothes_dataset/red_dress/4edafec8d798760fdba6892f0b23880225bfc393.jpg  
      inflating: ./clothes_dataset/red_dress/4edf5d16be7ed1e3d72e868f8679e80b0ba1644b.jpg  
      inflating: ./clothes_dataset/red_dress/4fd950bc0bdf39c701babefb517eca5614ce7eff.jpg  
      inflating: ./clothes_dataset/red_dress/4fe70740930b1b4a86bed817ccc03229bbd55784.jpg  
      inflating: ./clothes_dataset/red_dress/4fec8524b06757d5fd92758c812b194c006f611d.jpg  
      inflating: ./clothes_dataset/red_dress/5020c4fe1e4d515dddee580adf55d8bbd88e964f.jpg  
      inflating: ./clothes_dataset/red_dress/505b37413fef3752f2522012af4537abf2c12108.jpg  
      inflating: ./clothes_dataset/red_dress/505f12b36e522374a5593d25b2df076abbc2904b.jpg  
      inflating: ./clothes_dataset/red_dress/512876e5a35bdedd12ab88d821881b5cb949eb7d.jpg  
      inflating: ./clothes_dataset/red_dress/519ff8615aa732037ab6fc03874ea324945c8142.jpg  
      inflating: ./clothes_dataset/red_dress/51c4e41ce83d18143d537a49132cf1af40a47900.jpg  
      inflating: ./clothes_dataset/red_dress/51ec23389f29c649fc433aad99965158476a017e.jpg  
      inflating: ./clothes_dataset/red_dress/52062fbfc09bdee795576444852fb129e1b4d2de.jpg  
      inflating: ./clothes_dataset/red_dress/520e65f29b84428428900f33df834cb6981dd054.jpg  
      inflating: ./clothes_dataset/red_dress/52440ba40536a2038333d9a1cfb7d167f795ca90.jpg  
      inflating: ./clothes_dataset/red_dress/527eb9f449811e5c145b1414ec7f080277fc5c88.jpg  
      inflating: ./clothes_dataset/red_dress/53194e704718708b9bdfb3cfdab4a262b1ad8597.jpg  
      inflating: ./clothes_dataset/red_dress/5350b56eba4d7f0e6d171cbb238a40953ecc4e41.jpg  
      inflating: ./clothes_dataset/red_dress/53745e9157b4b81a054b4cc1f28647c4ba9a6270.jpg  
      inflating: ./clothes_dataset/red_dress/53a5bf0907d34daebbb20e1571086f8224bbfa10.jpg  
      inflating: ./clothes_dataset/red_dress/53b07d20a8bb44899ce4e7ed9267abbfe45e34dc.jpg  
      inflating: ./clothes_dataset/red_dress/53b298f1c2634cbfe2d11bb98fa7ee5e6e7c111f.jpg  
      inflating: ./clothes_dataset/red_dress/53e66c3b4e3ab7f0e31fc908b1c91fb9e3101665.jpg  
      inflating: ./clothes_dataset/red_dress/542460d73c362d52e5bd31c99fc89e31e884934b.jpg  
      inflating: ./clothes_dataset/red_dress/54f7b9a8a7e0e9b6d076cca319fc69fc127d3df6.jpg  
      inflating: ./clothes_dataset/red_dress/556c7b7bcab6041229de3bb22f5387edf365c999.jpg  
      inflating: ./clothes_dataset/red_dress/556cc12cb60f6c354e4f72e408fd7a00c3454192.jpg  
      inflating: ./clothes_dataset/red_dress/55a58afea5456f25318507bfcfbcf9f6cdd3d547.jpg  
      inflating: ./clothes_dataset/red_dress/55f27f0dcf0288dd0b2d6322c6cf6ae3e6c208b4.jpg  
      inflating: ./clothes_dataset/red_dress/5691ad37d04180637edd4fcfb4ca4f5dfbf0de70.jpg  
      inflating: ./clothes_dataset/red_dress/572273ce53eac568f27633d090a5b0e0123d3c9c.jpg  
      inflating: ./clothes_dataset/red_dress/57432d877e465d1d0d1d52a4aaba19039f004c21.jpg  
      inflating: ./clothes_dataset/red_dress/57707ed1cdb5d00a2d766557c2315f82f4ce202a.jpg  
      inflating: ./clothes_dataset/red_dress/57cd9709dbd109d3289287f6e59f59caca2aa606.jpg  
      inflating: ./clothes_dataset/red_dress/5853a4681be19c1bf41d82493321405b40efeccf.jpg  
      inflating: ./clothes_dataset/red_dress/58c9ecbbe8e6deb22e0de162d874262bff132b78.jpg  
      inflating: ./clothes_dataset/red_dress/58d9a73ad8c834691527b900d6bb0b86415341c7.jpg  
      inflating: ./clothes_dataset/red_dress/58ec7dd611d4c80ab402e47c5786b3d4cf2d6899.jpg  
      inflating: ./clothes_dataset/red_dress/5953061cf064f3a47464ba8326c37049ac1a0df1.jpg  
      inflating: ./clothes_dataset/red_dress/595c3dcd3f4b4afa73a3568a2c6f4289632e5f0c.jpg  
      inflating: ./clothes_dataset/red_dress/597c7be036db55e0fa16fe216d4c21621b90e1b3.jpg  
      inflating: ./clothes_dataset/red_dress/5a020d31d1d86986ac664498658de3cf7f169be5.jpg  
      inflating: ./clothes_dataset/red_dress/5a05a362b6d32320d203adfc1da6fd3718dd13b6.jpg  
      inflating: ./clothes_dataset/red_dress/5a36d78cf9cc91621bfab0f1dc331be66a5b288b.jpg  
      inflating: ./clothes_dataset/red_dress/5a4209d9f294001bbceb126be8d06f78542a0db5.jpg  
      inflating: ./clothes_dataset/red_dress/5ac1819208c42725a241460591512e78a33afe4f.jpg  
      inflating: ./clothes_dataset/red_dress/5af387756fc19d2f969edd3e9ea2f7ed73494239.jpg  
      inflating: ./clothes_dataset/red_dress/5b0b80784027a2283ad6002969d49667d31dd1ae.jpg  
      inflating: ./clothes_dataset/red_dress/5b3caf9b8d4aeec5002e59cd7017babab2365ee8.jpg  
      inflating: ./clothes_dataset/red_dress/5c85886762aa72cf9abe05758cfb6cccb846d374.jpg  
      inflating: ./clothes_dataset/red_dress/5d04e01546f9d5be90c0017b12e8abcdbe05887a.jpg  
      inflating: ./clothes_dataset/red_dress/5d06497714123982314af732c1343b4fcb668895.jpg  
      inflating: ./clothes_dataset/red_dress/5e80d740092b40c752e9f71051986c2a09e7f1eb.jpg  
      inflating: ./clothes_dataset/red_dress/5f072d5cd1fcdea88a6c042206aa1a819e6de671.jpg  
      inflating: ./clothes_dataset/red_dress/5faa34e01b29d964db6f9a30571abed6a4c776dc.jpg  
      inflating: ./clothes_dataset/red_dress/60509c0de66dc80e8893a9bf31caa0f6eb80cd7e.jpg  
      inflating: ./clothes_dataset/red_dress/608055788987a7b49f7fdb50592fb98d4543bcc5.jpg  
      inflating: ./clothes_dataset/red_dress/6182f37b819c45cd9e517b969091c43add2b83c5.jpg  
      inflating: ./clothes_dataset/red_dress/618b7c9101fedc6ca2e7db75b4053861d03c2cba.jpg  
      inflating: ./clothes_dataset/red_dress/61b425ab1202a11be6ddaff960d7a70439e56d6d.jpg  
      inflating: ./clothes_dataset/red_dress/6317257f8e16416b614d7693a9055968f96316f6.jpg  
      inflating: ./clothes_dataset/red_dress/6426ed98a044906ce18604f4bbf3fa5a0416ecd2.jpg  
      inflating: ./clothes_dataset/red_dress/642d006670f45465ff6aabd025844355aaa41868.jpg  
      inflating: ./clothes_dataset/red_dress/646653a905f045e4a05c4d3a6d221ab905e59419.jpg  
      inflating: ./clothes_dataset/red_dress/6468010a7340abad775957b4fc96f0e595d20fff.jpg  
      inflating: ./clothes_dataset/red_dress/651ce54897b409cda86dff62ff2c5276a4c8bbc6.jpg  
      inflating: ./clothes_dataset/red_dress/651ff25eeb3da3672d264d17c74508748bdbf929.jpg  
      inflating: ./clothes_dataset/red_dress/6545f66de54f8089f5b8cc0447ac170997485dd3.jpg  
      inflating: ./clothes_dataset/red_dress/664bce495347079e4d96240a76bfe634caf2904e.jpg  
      inflating: ./clothes_dataset/red_dress/667544ecf6164ef3dc44efd967cdd1b8ded5cd0b.jpg  
      inflating: ./clothes_dataset/red_dress/6692d7c504ddb3c9da8fa166d1d7e346af6f9d1e.jpg  
      inflating: ./clothes_dataset/red_dress/673addc36d1dcb7f54ba6ad54b9ab5ad31adf3ca.jpg  
      inflating: ./clothes_dataset/red_dress/67a500aef982bf6620c7f636e39b2e52c18c5389.jpg  
      inflating: ./clothes_dataset/red_dress/67b4c174b7078d2c2836803f85275e4f9a249eb3.jpg  
      inflating: ./clothes_dataset/red_dress/67b6a26e9c091a3fbfa516277f95a0eb2a42b213.jpg  
      inflating: ./clothes_dataset/red_dress/682e17c7a21ec656a4f734ec1b2eeef665d46153.jpg  
      inflating: ./clothes_dataset/red_dress/683923a0af0b8ddbebbc09683a4d567570b10d2a.jpg  
      inflating: ./clothes_dataset/red_dress/68e65336402bcf0269e1b2d24c1a94844cda74b2.jpg  
      inflating: ./clothes_dataset/red_dress/69078937b02ad7b75c5df93f5cde5d7a1624643d.jpg  
      inflating: ./clothes_dataset/red_dress/69d5dc91e97e27874f5170c6cbaded7e910028aa.jpg  
      inflating: ./clothes_dataset/red_dress/6a231c6832b23e19bd928d4047a7866b1c65cd08.jpg  
      inflating: ./clothes_dataset/red_dress/6a9071b001b232563ca7122c9441516761a902bf.jpg  
      inflating: ./clothes_dataset/red_dress/6ad0a46b833da137acedc12f16a97c9efa5c0889.jpg  
      inflating: ./clothes_dataset/red_dress/6afbc153ff5504b348a17dd26ce58e50aedee46e.jpg  
      inflating: ./clothes_dataset/red_dress/6b2c26b1a5c758384d26bc4aff45e208d22f4eb4.jpg  
      inflating: ./clothes_dataset/red_dress/6b92bc9975c8c03ce9d5dae0becefd1621bcb8b2.jpg  
      inflating: ./clothes_dataset/red_dress/6b9463e98cce2f241445421fe69137c0f93787d4.jpg  
      inflating: ./clothes_dataset/red_dress/6cf0a1e0b1d61d5cf9fa87a1f3fe03e1564c88cf.jpg  
      inflating: ./clothes_dataset/red_dress/6db0c991b2c17bae2b4c1f266b05e7fbc3811499.jpg  
      inflating: ./clothes_dataset/red_dress/6e059564af715485574a176f5b4c6d50066ee6b6.jpg  
      inflating: ./clothes_dataset/red_dress/6e7b32491a3a917eb307c2aff0aeb474111e4d3b.jpg  
      inflating: ./clothes_dataset/red_dress/6e878d7d3c2836401284cd29165c6d03e561fc00.jpg  
      inflating: ./clothes_dataset/red_dress/6f5f0ff9fc4573d658f3ea39d246832c08474dee.jpg  
      inflating: ./clothes_dataset/red_dress/6fcf4a2e208ee0905a900cc1a816ddcb564c930c.jpg  
      inflating: ./clothes_dataset/red_dress/7039b55e3f539a3a3e44d2eb7dacdee5c202c2d2.jpg  
      inflating: ./clothes_dataset/red_dress/703b3f8c05498e93921a6645f1f23f79ea9eeee3.jpg  
      inflating: ./clothes_dataset/red_dress/70556bb9f453d7e8ede1aa635b1adb89d0a639ee.jpg  
      inflating: ./clothes_dataset/red_dress/707d083176c05de6ad3cec863140f4f2c3b37648.jpg  
      inflating: ./clothes_dataset/red_dress/70b9d3f569b5148e8b7c179ea27763a8f56625bd.jpg  
      inflating: ./clothes_dataset/red_dress/71e562ebbe1a0ec545b4b65da3a8037ad74947f7.jpg  
      inflating: ./clothes_dataset/red_dress/720dd1006977ca88012eee3078fd35b5e403cc07.jpg  
      inflating: ./clothes_dataset/red_dress/7217d69241a0df639f071e6e0bd9c52c11be8f32.jpg  
      inflating: ./clothes_dataset/red_dress/721c3fdeb2a86272f03a0407586e2b13c746ff47.jpg  
      inflating: ./clothes_dataset/red_dress/7235bd63152a550be525665006aeb48f6d42f1bb.jpg  
      inflating: ./clothes_dataset/red_dress/728ebcbc6cbfa7677e88e65e74bbb807e227bbc1.jpg  
      inflating: ./clothes_dataset/red_dress/729e134bcedb6ac7468bb7714a62e62de5881559.jpg  
      inflating: ./clothes_dataset/red_dress/72af0395ee92e866813197a52a431721ce7c1798.jpg  
      inflating: ./clothes_dataset/red_dress/72dfe79d3947de770067b455852b4436a741cc78.jpg  
      inflating: ./clothes_dataset/red_dress/72eb926a315edd82ee1d35ca2d3b50a10074a19f.jpg  
      inflating: ./clothes_dataset/red_dress/72eea1a1bbaef6e0a22082cba984a51aa540619e.jpg  
      inflating: ./clothes_dataset/red_dress/7355b27991cb2bc773bb3096b373a3264ce0d4b5.jpg  
      inflating: ./clothes_dataset/red_dress/73a34c30f8fbd7c369997c181b1aee5e94b60d03.jpg  
      inflating: ./clothes_dataset/red_dress/73aeea7f1f7157d7ad842ed2e30165b32674f81d.jpg  
      inflating: ./clothes_dataset/red_dress/73d0542a6b1e4d1d86cf2b561abc0417e7bb9b12.jpg  
      inflating: ./clothes_dataset/red_dress/745c559c777da3797f41e46a0406a39edd5a2c0f.jpg  
      inflating: ./clothes_dataset/red_dress/74775babe90096bbae9bd33036946a27156b5fb1.jpg  
      inflating: ./clothes_dataset/red_dress/74a4ddd31705012a070dd8724a7b00e8192e73cc.jpg  
      inflating: ./clothes_dataset/red_dress/74c71aa13eaad5a10cf644a1cdc3d8e1f73718ed.jpg  
      inflating: ./clothes_dataset/red_dress/74e2e5d243461196e654704bf8883f416fa9842b.jpg  
      inflating: ./clothes_dataset/red_dress/75036c8875370d15f3af96c6006faa1e8996da7c.jpg  
      inflating: ./clothes_dataset/red_dress/75a7c07b4e5a350071f7b3390b91045cb958aa02.jpg  
      inflating: ./clothes_dataset/red_dress/764e626f8577c27c85b3950dea09fdd90575d30b.jpg  
      inflating: ./clothes_dataset/red_dress/769981ad28e44ebf13d8866e59ffa3dd56716cc6.jpg  
      inflating: ./clothes_dataset/red_dress/76e7118b8e9736b7caf562c3148deb9bb48550dc.jpg  
      inflating: ./clothes_dataset/red_dress/779dddf8d0b1b5983c61374e112b581223a592fc.jpg  
      inflating: ./clothes_dataset/red_dress/77e609d810f864f3a5988dd3fb132295500e86a1.jpg  
      inflating: ./clothes_dataset/red_dress/789202a1dbc117e634670d868734b5bbf9c1c078.jpg  
      inflating: ./clothes_dataset/red_dress/791f75ae2513a3065c96ae0302b05355af508d72.jpg  
      inflating: ./clothes_dataset/red_dress/79ac5189cada8c464825e609bc637f14151a89e4.jpg  
      inflating: ./clothes_dataset/red_dress/7a121905c4dacd3d35cad3756284953efbb9276c.jpg  
      inflating: ./clothes_dataset/red_dress/7a2f7e7fe1eeb960823c025bfeb69e44695df13a.jpg  
      inflating: ./clothes_dataset/red_dress/7b2b7219f33aa727a72a19b1aa577faa9f5c3447.jpg  
      inflating: ./clothes_dataset/red_dress/7bd7a0404a3229e65fa711d9b9bee7c6f9477057.jpg  
      inflating: ./clothes_dataset/red_dress/7c600f2f8c2ebe4132cb5d83bf1fbde6e6144135.jpg  
      inflating: ./clothes_dataset/red_dress/7c6e054e0bb9c62034a98413b301880ff059a8ca.jpg  
      inflating: ./clothes_dataset/red_dress/7c71834a56816c95bfb3d9d5fb4805586f6d2686.jpg  
      inflating: ./clothes_dataset/red_dress/7c7e708d04e974cd113fd6fb2bf595fc1a041229.jpg  
      inflating: ./clothes_dataset/red_dress/7d042891a4a5095a56ee245c433c5001d5b3b881.jpg  
      inflating: ./clothes_dataset/red_dress/7d1802fe5796467369881d5f890460c9676f347a.jpg  
      inflating: ./clothes_dataset/red_dress/7d48135ea7eba15191d3b6da21c2e9ef19ec197e.jpg  
      inflating: ./clothes_dataset/red_dress/7d7047a4ca9b78672d319a039db665f21f0c3dc0.jpg  
      inflating: ./clothes_dataset/red_dress/7d83d12fc53749bb4cff685e2e77efb45d0a39cd.jpg  
      inflating: ./clothes_dataset/red_dress/7dae556b1ac1b0ba6edc1e4ecb83955dfecd572d.jpg  
      inflating: ./clothes_dataset/red_dress/7e37ffefa47489b4e617effa772921e84345242f.jpg  
      inflating: ./clothes_dataset/red_dress/7f331bdfdb4b965035248b97b93202db0dba3502.jpg  
      inflating: ./clothes_dataset/red_dress/7f427c3e1f6d29cb0b34307f78894df7cf2d3866.jpg  
      inflating: ./clothes_dataset/red_dress/7fab13dcd4287e44c510661eda50e6e2b8ae2e85.jpg  
      inflating: ./clothes_dataset/red_dress/8019252d8c29fd94f1e27a2bc555223d5263019b.jpg  
      inflating: ./clothes_dataset/red_dress/8050eb25e98a1b38f0693e0b5a28067bcc1534f5.jpg  
      inflating: ./clothes_dataset/red_dress/80dcd2c108752fa50e4712acf3f3d38610fb43b4.jpg  
      inflating: ./clothes_dataset/red_dress/81a39e2f728a4dc21f1215d2326aeda72f465ae8.jpg  
      inflating: ./clothes_dataset/red_dress/81d38efa588a5eeb765d957ecb1d5bcc6c103e50.jpg  
      inflating: ./clothes_dataset/red_dress/81e9722effc34d807c6b7b4529de2470979fcc55.jpg  
      inflating: ./clothes_dataset/red_dress/8290583a60b59e394da570ccd70fc7d1d54cf9cd.jpg  
      inflating: ./clothes_dataset/red_dress/82afa18718cf38a776584af01fe27e04486211f7.jpg  
      inflating: ./clothes_dataset/red_dress/82c2d2fef53171bb36fd5fe427fccf1cfef0face.jpg  
      inflating: ./clothes_dataset/red_dress/82c87b740179fef667558cfa09193ef6e48b1146.jpg  
      inflating: ./clothes_dataset/red_dress/832036911c3d8a27760107c29045c1f4ed264305.jpg  
      inflating: ./clothes_dataset/red_dress/8348ccf0d600be95d95094b9d1d5141bb48cc8f8.jpg  
      inflating: ./clothes_dataset/red_dress/836e62890b76965f928fd34f2ef8cdf79ca47c85.jpg  
      inflating: ./clothes_dataset/red_dress/8374f59d0e25aa5c703520a4b5d8adef4bb249ee.jpg  
      inflating: ./clothes_dataset/red_dress/838cae0f44d6966dcd0d236959f8cf427d72c8a5.jpg  
      inflating: ./clothes_dataset/red_dress/83c4b79ca7de1796e400b32f044c194ed5d3467d.jpg  
      inflating: ./clothes_dataset/red_dress/83c896e27ebc8252009f9a516647232e8070cfc8.jpg  
      inflating: ./clothes_dataset/red_dress/83dc4d23b1311f5f945239d2c7e109a9c652e104.jpg  
      inflating: ./clothes_dataset/red_dress/8434cf5dfab5c43f40acc8e73bc06b10a0b75874.jpg  
      inflating: ./clothes_dataset/red_dress/8442f47c96943db7325786266f406ab5cd85dc2e.jpg  
      inflating: ./clothes_dataset/red_dress/84c1a4377272e87cb9e19d82c764818891226a2f.jpg  
      inflating: ./clothes_dataset/red_dress/850cfab6b83dd79b02e7e15ad1813cd93dc2db59.jpg  
      inflating: ./clothes_dataset/red_dress/85160ff33a85fbb7ff6e9859389f18be63c61a8d.jpg  
      inflating: ./clothes_dataset/red_dress/8577d1bdc847584e9e7c48bafba8ba9ccf8a6266.jpg  
      inflating: ./clothes_dataset/red_dress/859354fa5f548c653d4a26ceec254633f4b2c722.jpg  
      inflating: ./clothes_dataset/red_dress/86281eb35b432aeffff86b7c3b4a2a861ed337e8.jpg  
      inflating: ./clothes_dataset/red_dress/863fb9b546adfa4c410d775a5555aa862e6c63b6.jpg  
      inflating: ./clothes_dataset/red_dress/86a3eb8dae6df8a49996bd19fb5e860b09799c3a.jpg  
      inflating: ./clothes_dataset/red_dress/86afa57afabec0378f555904cf19dd846a1c6c6d.jpg  
      inflating: ./clothes_dataset/red_dress/86c3be01f5672dc47c2fd97acfd9ef452a2cc4c1.jpg  
      inflating: ./clothes_dataset/red_dress/86c5bdf7d10945e7bb92ea75f9ca1686b01c09a5.jpg  
      inflating: ./clothes_dataset/red_dress/873b6b5fbda008b8d363b923f218d5f3ee926037.jpg  
      inflating: ./clothes_dataset/red_dress/8820b94444c4837b0556a5df1c1b3819c5026c39.jpg  
      inflating: ./clothes_dataset/red_dress/88c3b3ac7f5cc85349471d8323db0774f81919c1.jpg  
      inflating: ./clothes_dataset/red_dress/88fb1837a72c35b6335704f72bacee590504f43e.jpg  
      inflating: ./clothes_dataset/red_dress/89174dd138e77ca41ed884835318d774ed86f1dc.jpg  
      inflating: ./clothes_dataset/red_dress/899059eae6edce2339d8ef59cf62e82759f07ea8.jpg  
      inflating: ./clothes_dataset/red_dress/89db9cf974da198d3ce0fb1e6b38b25c9aa85c5a.jpg  
      inflating: ./clothes_dataset/red_dress/89fbfc13682a359267db94f9af70d9ff1537b71a.jpg  
      inflating: ./clothes_dataset/red_dress/8a0232c4b5f9a3224c070b32b52446c9cc1f679f.jpg  
      inflating: ./clothes_dataset/red_dress/8a191ca59ee809fec719f21960a3927c9181777a.jpg  
      inflating: ./clothes_dataset/red_dress/8a49ec6c4a58822d2e19a1b3830829e87409a61d.jpg  
      inflating: ./clothes_dataset/red_dress/8a639b35087697573da0848f945cb1f3e7dac5e6.jpg  
      inflating: ./clothes_dataset/red_dress/8afb24ac4d527868647e0af5f997a79ef0b48433.jpg  
      inflating: ./clothes_dataset/red_dress/8b14b50397772b6b6b39048c6366d0d491482455.jpg  
      inflating: ./clothes_dataset/red_dress/8bf361abfa9f4d238a027396b08ee9d2e10d65d9.jpg  
      inflating: ./clothes_dataset/red_dress/8c9ccd6c4e32b5027a184f63a3ebf112c1691ad0.jpg  
      inflating: ./clothes_dataset/red_dress/8d211ee0577ec11fe62e3aa5cda582f3c0113921.jpg  
      inflating: ./clothes_dataset/red_dress/8dd5296e22eb38d350bb1de0a26220b5f1e4bf03.jpg  
      inflating: ./clothes_dataset/red_dress/8dda59abb46a5acafe48bfa47fb36049402b222a.jpg  
      inflating: ./clothes_dataset/red_dress/8e2ec51cd8b173522c38bfdaac81d853698a0864.jpg  
      inflating: ./clothes_dataset/red_dress/8e3b92cf9991b00dd045d689b0fd894d9fa62ea1.jpg  
      inflating: ./clothes_dataset/red_dress/8e8814b0fe932fbca747b1148306a8235bc68b30.jpg  
      inflating: ./clothes_dataset/red_dress/8eeffccd2a7621393a641c1dd6a4bf698b705e3d.jpg  
      inflating: ./clothes_dataset/red_dress/8f97f82f0a440201c1750f29af2e28e69b8d75f5.jpg  
      inflating: ./clothes_dataset/red_dress/8fd152a8d1a75332b4f96800a97802a9877c63dc.jpg  
      inflating: ./clothes_dataset/red_dress/903ff85506fe572b51dc84d4266a574ddb5530ca.jpg  
      inflating: ./clothes_dataset/red_dress/907bd6a6e253d234a175ef3732aa362e5a8fc6d8.jpg  
      inflating: ./clothes_dataset/red_dress/90a3974e4666c144bf2acec4a6d37d8669385fa8.jpg  
      inflating: ./clothes_dataset/red_dress/90b4d431f09f176ba88265fa39aa77de8977a8b9.jpg  
      inflating: ./clothes_dataset/red_dress/90f19ce36d013b634c2e3f0451f03d8ffe0a8d30.jpg  
      inflating: ./clothes_dataset/red_dress/91934ca9a46033e8973875b684468950496645fe.jpg  
      inflating: ./clothes_dataset/red_dress/91bd966c5d0f4df4d86769cbfbc323b621984f33.jpg  
      inflating: ./clothes_dataset/red_dress/91c0844d6104e83a569cab554e6308ddccc89415.jpg  
      inflating: ./clothes_dataset/red_dress/921e5b129165bd98ca16d18aa8e2f551e856c162.jpg  
      inflating: ./clothes_dataset/red_dress/92f4dfcc67948c8c682b91ef82f4567bb3d9138e.jpg  
      inflating: ./clothes_dataset/red_dress/938e7eb6397bcde59c27e260327c3c74c6b41848.jpg  
      inflating: ./clothes_dataset/red_dress/93f8533e4f9a4254e7a198aee8d3bda401393c5b.jpg  
      inflating: ./clothes_dataset/red_dress/93fab51b26720e59ec0099369e53ffec713564f2.jpg  
      inflating: ./clothes_dataset/red_dress/940dcf0f366e71b4ddc729529b35f0aaaf1d416e.jpg  
      inflating: ./clothes_dataset/red_dress/942f4b20a8630ed7d999a58662ac85dd604ee972.jpg  
      inflating: ./clothes_dataset/red_dress/944135f9842f0aa2ea85c0db998b10ebf063240f.jpg  
      inflating: ./clothes_dataset/red_dress/9459dfb74e954c1652409085365aa36ab3e0ae7c.jpg  
      inflating: ./clothes_dataset/red_dress/948df9076253f8df974f395a3f5f696e071f2bd8.jpg  
      inflating: ./clothes_dataset/red_dress/9494224443cea785caaecf1357ec2831d681dbf4.jpg  
      inflating: ./clothes_dataset/red_dress/94bfb6f88154fd1142402795983a71f85e593664.jpg  
      inflating: ./clothes_dataset/red_dress/94e75ec8bc061fb80b06be0e437fb14fe89f08bb.jpg  
      inflating: ./clothes_dataset/red_dress/94f3269fc97ff61f07f8083aa8b82ced68b26925.jpg  
      inflating: ./clothes_dataset/red_dress/95133f504c3d6b8d693a5c70b38b1681f65ebb84.jpg  
      inflating: ./clothes_dataset/red_dress/962c284a0dfdf89e68684d55d1091795ecec8c2a.jpg  
      inflating: ./clothes_dataset/red_dress/9674cab4c2fd280304a8d2453e615e70a1c3f4c8.jpg  
      inflating: ./clothes_dataset/red_dress/96baf511ac987919c1b183225ed6d91bf92de7a0.jpg  
      inflating: ./clothes_dataset/red_dress/97173ce08cb0196a551890d61b0cb852000f2025.jpg  
      inflating: ./clothes_dataset/red_dress/97c99edebb20235fbe071ee775c267ce1dd4d883.jpg  
      inflating: ./clothes_dataset/red_dress/984e7cd0eef68538963c099960917a36b5e2a1c1.jpg  
      inflating: ./clothes_dataset/red_dress/9893f9cc2ee8dcf738af70edd0f3750908da5758.jpg  
      inflating: ./clothes_dataset/red_dress/98a6d49a665466c3475547e288d1553a51c4b477.jpg  
      inflating: ./clothes_dataset/red_dress/98c23ef1b1fd951deffbb8882910d4acea7ed824.jpg  
      inflating: ./clothes_dataset/red_dress/98dafbedcfb02867177de17b907fd89727743094.jpg  
      inflating: ./clothes_dataset/red_dress/99259bfa922d1b0c9f5c98b4c0aabbbaaad9d843.jpg  
      inflating: ./clothes_dataset/red_dress/9930e560f3e543c3f4018615994263340dd0348f.jpg  
      inflating: ./clothes_dataset/red_dress/9933ed3950e1c509ac6d7af4f69ec2c2a0349046.jpg  
      inflating: ./clothes_dataset/red_dress/99424e04699d6d7838d1bd2c796e01b1d703fde5.jpg  
      inflating: ./clothes_dataset/red_dress/9af91a1a06121132296dba9c8c79531f004336fd.jpg  
      inflating: ./clothes_dataset/red_dress/9bfb9fc6d5f7ba79fa302dcc1f6ab5ae8da8cec6.jpg  
      inflating: ./clothes_dataset/red_dress/9cd72e348d781e5e55004cc7db08184810306540.jpg  
      inflating: ./clothes_dataset/red_dress/9d03d82cbfca825a15e53e5a13ffdef778658313.jpg  
      inflating: ./clothes_dataset/red_dress/9ec276859e406b33f661a3a93400884144272283.jpg  
      inflating: ./clothes_dataset/red_dress/9ecdfd0f491d0c11666d35c0a631100a3f40f983.jpg  
      inflating: ./clothes_dataset/red_dress/9f1c56c44319d48ac4fa920a20baef3faa6ce49f.jpg  
      inflating: ./clothes_dataset/red_dress/a00afcb0b8315c737baf17025d7fe37bca933e98.jpg  
      inflating: ./clothes_dataset/red_dress/a04772e6d8bed2b089044cbe075091240ab0aedf.jpg  
      inflating: ./clothes_dataset/red_dress/a06deb421ccab55c42f6c4a74211502a0076758f.jpg  
      inflating: ./clothes_dataset/red_dress/a10b1b7fe08b61239fb8966ebe357ec3ecd34daa.jpg  
      inflating: ./clothes_dataset/red_dress/a11a2a4873c6bcb92e4938d5079f3236d13ccabd.jpg  
      inflating: ./clothes_dataset/red_dress/a16bdbd25f4af14d8cc4e0e69a7c0f1f15244e14.jpg  
      inflating: ./clothes_dataset/red_dress/a181652e753ad7cc863081c07d2a41c7fdca1453.jpg  
      inflating: ./clothes_dataset/red_dress/a1f2a05aebea4e68fbf54d9631c878d0afbb4706.jpg  
      inflating: ./clothes_dataset/red_dress/a1f4347a6ceec11e4044ad57ff5255f1973a94bb.jpg  
      inflating: ./clothes_dataset/red_dress/a23589388ce6514f66850550c1b27f4d1dd2e88a.jpg  
      inflating: ./clothes_dataset/red_dress/a250535ab38ca6d1f27fb832305305d525197eb3.jpg  
      inflating: ./clothes_dataset/red_dress/a252765346042a134252e504b7babfed5bc9152d.jpg  
      inflating: ./clothes_dataset/red_dress/a2b5117f002e7558e7804f898a077fc10bcf7ab6.jpg  
      inflating: ./clothes_dataset/red_dress/a2cf994126702ed5afec5cb53a576a0e98a6d0b6.jpg  
      inflating: ./clothes_dataset/red_dress/a2e0089adcb05845a4762e9865ce6f42d1866138.jpg  
      inflating: ./clothes_dataset/red_dress/a3af67a6ef2b339a9b5416be68b0b5c7acfae3b8.jpg  
      inflating: ./clothes_dataset/red_dress/a42deeb54f6487b74a7c79584ad0696990baf720.jpg  
      inflating: ./clothes_dataset/red_dress/a44c83fd1d4df807f1f7a2a1e9d5de9f2a195b30.jpg  
      inflating: ./clothes_dataset/red_dress/a48cef70da7230c1b2c1878cc2986a50c4aa9756.jpg  
      inflating: ./clothes_dataset/red_dress/a4decb48252850b806fd5de2dc2fafe435e79e81.jpg  
      inflating: ./clothes_dataset/red_dress/a556cc14e89659a2d8f8060d96ca36f619be8203.jpg  
      inflating: ./clothes_dataset/red_dress/a5ad66fb06af5211786a4fd13399e5452696ccde.jpg  
      inflating: ./clothes_dataset/red_dress/a6e084612ff1e1d2e4a4255e12f1ed3852d507ad.jpg  
      inflating: ./clothes_dataset/red_dress/a77f3bd9ebc3da2435f78f5751f29fd4e99157f8.jpg  
      inflating: ./clothes_dataset/red_dress/a79da05f27d815e83b103d56510ff0f6a6de6f4a.jpg  
      inflating: ./clothes_dataset/red_dress/a7ef15b19c489620d0a4bc0fc9871f029e6193aa.jpg  
      inflating: ./clothes_dataset/red_dress/a7fa9fe294e1d4bd921ca1b5f501e48f549da692.jpg  
      inflating: ./clothes_dataset/red_dress/a82cfec5e77fe54425c4b957c8125851acae27ec.jpg  
      inflating: ./clothes_dataset/red_dress/a84ccb6f210256db61b05f0eed65304653226740.jpg  
      inflating: ./clothes_dataset/red_dress/a939e6698bbdf3df77781d6a732c113714ff1224.jpg  
      inflating: ./clothes_dataset/red_dress/a93cc1837233bf1136748e8cc7e130fee508c302.jpg  
      inflating: ./clothes_dataset/red_dress/a94d9ea29ba7e98ca6e84666c8e6db5e0beb16c4.jpg  
      inflating: ./clothes_dataset/red_dress/a96c6e13c0bebaabd873dc6d93a83840bb1f72ef.jpg  
      inflating: ./clothes_dataset/red_dress/a984e179a98790cbe9d6d83a129e3e8adb1fe94c.jpg  
      inflating: ./clothes_dataset/red_dress/a9941242a865dd16c54ff834f49885b5ea838144.jpg  
      inflating: ./clothes_dataset/red_dress/a9bc946d7cbb7f6058d2224a02ceef0de123fc60.jpg  
      inflating: ./clothes_dataset/red_dress/aa3b75645130ef33c64acadc137fbceac9a4f352.jpg  
      inflating: ./clothes_dataset/red_dress/aacdb18b7694e1c28e52b91be43cb025b538ed00.jpg  
      inflating: ./clothes_dataset/red_dress/aafa7b5e529b18d66360489a3572209cd4f18f9e.jpg  
      inflating: ./clothes_dataset/red_dress/ab1c91359fc493a803a2669bdf4d62f1437802dd.jpg  
      inflating: ./clothes_dataset/red_dress/ab2e00c18e3ec773646ef4eb9c29d070c6e075fc.jpg  
      inflating: ./clothes_dataset/red_dress/ab8e18a47f3cac5f5d48e4576ee9f42fef2897e1.jpg  
      inflating: ./clothes_dataset/red_dress/abd1dd9e3c0b2e6c0d5edb8c3fbe130e93e3569f.jpg  
      inflating: ./clothes_dataset/red_dress/abf1d6db415e09bdbf56c048ce8bccaad3cf77ef.jpg  
      inflating: ./clothes_dataset/red_dress/acda6fb03af51ddfca009e32f24254374d24c397.jpg  
      inflating: ./clothes_dataset/red_dress/acf0f567cc74cb6bda07b79be5fa1c7618317bd6.jpg  
      inflating: ./clothes_dataset/red_dress/ad20e4a224cfe437d657e8ee6bab4c4a4360fda7.jpg  
      inflating: ./clothes_dataset/red_dress/ad629e089bc4771872c48dada2a2a9b2a663a722.jpg  
      inflating: ./clothes_dataset/red_dress/adcdc4cb53cad73a6138f34312b4a2105ce56cb8.jpg  
      inflating: ./clothes_dataset/red_dress/adcf868cdb7f8e00a6994ed47e4e1adb2d6a2b1b.jpg  
      inflating: ./clothes_dataset/red_dress/add0dc66c72208fbec17693bb4a6707fad78833f.jpg  
      inflating: ./clothes_dataset/red_dress/ae3fc8543d94268a67e8f334decb2316b12adc97.jpg  
      inflating: ./clothes_dataset/red_dress/ae7614ea71b1d54e50eda683a8a9396140b2a8b8.jpg  
      inflating: ./clothes_dataset/red_dress/ae97babc61107d4fd7a27f75f3ea550ded0b5006.jpg  
      inflating: ./clothes_dataset/red_dress/aed439502bea6aab87bb0510646a7ff877b7792a.jpg  
      inflating: ./clothes_dataset/red_dress/af27563e9957dea79babeedd3aab22abec110b7f.jpg  
      inflating: ./clothes_dataset/red_dress/af6e426435680d11688ec951af3873e7b3371fad.jpg  
      inflating: ./clothes_dataset/red_dress/afa58e64246c242360db7e2134ce81a3e7d71171.jpg  
      inflating: ./clothes_dataset/red_dress/b04b501a2c14859ef487e87983a754e214cb8a4b.jpg  
      inflating: ./clothes_dataset/red_dress/b04bcbae214230683b876be89b98ccd4e2ad7afc.jpg  
      inflating: ./clothes_dataset/red_dress/b068b416cf504e5635ce6e7155636cb3de74f764.jpg  
      inflating: ./clothes_dataset/red_dress/b0d187d343dd496ef60421397971585b17b34930.jpg  
      inflating: ./clothes_dataset/red_dress/b113d66ef01249dcadb6ab63ed57f3b9c3040147.jpg  
      inflating: ./clothes_dataset/red_dress/b1812657d4a9859ff4a2db502191fcae2f4d6117.jpg  
      inflating: ./clothes_dataset/red_dress/b193d627a92a9e89e11c36721a62df5e3bc2e32b.jpg  
      inflating: ./clothes_dataset/red_dress/b1ad89f59b51fa126043eaa8f86906bc7325e6c7.jpg  
      inflating: ./clothes_dataset/red_dress/b21641079c2d0115123de4db0316a11a50f1d6b0.jpg  
      inflating: ./clothes_dataset/red_dress/b23038de268e2d5e6c7a5da6971f886f673da43d.jpg  
      inflating: ./clothes_dataset/red_dress/b2f9b9fbf266c488bea57901ca1ac30fb6eebfd2.jpg  
      inflating: ./clothes_dataset/red_dress/b3246f397a103a53acba55e0f2dcfd3588fc7680.jpg  
      inflating: ./clothes_dataset/red_dress/b3799c817ce0dc777accfebb46430ca2f7455900.jpg  
      inflating: ./clothes_dataset/red_dress/b39c70f1df126ae39fe1aad333165ec6066ab447.jpg  
      inflating: ./clothes_dataset/red_dress/b465fab127d64f22da2be14bd9201c96975940da.jpg  
      inflating: ./clothes_dataset/red_dress/b4f5290095dd43b69be9b73fbcb72e7766289e8f.jpg  
      inflating: ./clothes_dataset/red_dress/b52c4ad5a1c3b1ed8669877666f5ef7cde97abdb.jpg  
      inflating: ./clothes_dataset/red_dress/b550200a1b0d2e98de6e77b0db3fa186de89ab81.jpg  
      inflating: ./clothes_dataset/red_dress/b5a509200b3ecad64733dca340d55b17a8d1d801.jpg  
      inflating: ./clothes_dataset/red_dress/b5e37a772cf753a53bba7e4d1caa798b15144487.jpg  
      inflating: ./clothes_dataset/red_dress/b62886131acd841600be61c1c29f7d22ceea6163.jpg  
      inflating: ./clothes_dataset/red_dress/b6a41449452260ebde82ba9e977a53e3f48e0fb7.jpg  
      inflating: ./clothes_dataset/red_dress/b6ea43d04d14b0c2ea26800fa5222e257d47413b.jpg  
      inflating: ./clothes_dataset/red_dress/b6f07ee2e4b5680aa68b589c71d6483fdd762298.jpg  
      inflating: ./clothes_dataset/red_dress/b73ac36037f4d3c9f0bdce2d08b8f55aac2e815a.jpg  
      inflating: ./clothes_dataset/red_dress/b742e986ce5b7266bc01da9cec66a47e72fd0ef9.jpg  
      inflating: ./clothes_dataset/red_dress/b81e385b7b17793880716a4919d0c859842317cd.jpg  
      inflating: ./clothes_dataset/red_dress/b83fde602971f117d36332b90f350d2075de4e11.jpg  
      inflating: ./clothes_dataset/red_dress/b871401eac2f13d49f59b1bbb51bd68ecb46eb88.jpg  
      inflating: ./clothes_dataset/red_dress/b88aa770691a62d0be5cddf6b4d877d1e10eb39e.jpg  
      inflating: ./clothes_dataset/red_dress/b8964f1706e0057735811ae76567a8193600800c.jpg  
      inflating: ./clothes_dataset/red_dress/b90f79307a31051c9e69a064253fedafc292053f.jpg  
      inflating: ./clothes_dataset/red_dress/b953aed564fa8068b18eb98c5972e75f0d7e1ce2.jpg  
      inflating: ./clothes_dataset/red_dress/b98389df3b5b7eb40eb18cf650ee0b09e58c82ff.jpg  
      inflating: ./clothes_dataset/red_dress/ba769b06649b8fdaca4f230061adafc2c5455ec2.jpg  
      inflating: ./clothes_dataset/red_dress/bae8b43848000ef0e82c856db6cea1fedbf4bdb9.jpg  
      inflating: ./clothes_dataset/red_dress/baf791f6da5b787130226d5783e8b713d4ed10c9.jpg  
      inflating: ./clothes_dataset/red_dress/bb130b5050e3895afde1c6146883a18fc2dba925.jpg  
      inflating: ./clothes_dataset/red_dress/bb2af2b61740f6e2232b0389f1c5bf43bc3dc70e.jpg  
      inflating: ./clothes_dataset/red_dress/bb5fc4b7282d7d6e9e60dfc07a82057a2f61b04e.jpg  
      inflating: ./clothes_dataset/red_dress/bb81666a8d4bb63d400a231a634a22209dc9672e.jpg  
      inflating: ./clothes_dataset/red_dress/bdaeea66cba48864f2a084f1a0a3dc92bed0dcb8.jpg  
      inflating: ./clothes_dataset/red_dress/bdd2b793e233fe98cccc3b1823024c10c9495f02.jpg  
      inflating: ./clothes_dataset/red_dress/bdd9c90903fffc9720626ea663a621776d00543a.jpg  
      inflating: ./clothes_dataset/red_dress/be03989457840ad4e2b2aa33f62dfbe4389b1a13.jpg  
      inflating: ./clothes_dataset/red_dress/be2ab4fb7ae511493ce10d90ed329d73d17d1c6b.jpg  
      inflating: ./clothes_dataset/red_dress/bf16790cff4ab670eb43e2f517b0038d8c8fa6ad.jpg  
      inflating: ./clothes_dataset/red_dress/bf7acf34b8d296f8afda787387746115a2290485.jpg  
      inflating: ./clothes_dataset/red_dress/bf857e585b5bb919d7c45ceef4fd6a4aa4e193bd.jpg  
      inflating: ./clothes_dataset/red_dress/c074e1c00f0a8b97bc568a4fd96a0ef9e5f8d534.jpg  
      inflating: ./clothes_dataset/red_dress/c09a43a0e332120acc86443943adb8b30e679df4.jpg  
      inflating: ./clothes_dataset/red_dress/c129317a82d9d1e84cf3b99c1f1dd3bbbe665beb.jpg  
      inflating: ./clothes_dataset/red_dress/c177ea09c020e65229c3c3d3a27e76efa8fac87f.jpg  
      inflating: ./clothes_dataset/red_dress/c1cfa91feb5122286c8a02b71be0e7cead51ef7d.jpg  
      inflating: ./clothes_dataset/red_dress/c23764f2d23832f7d2e65169694dcd182e14bfcd.jpg  
      inflating: ./clothes_dataset/red_dress/c2465e45df6d4c84588461a9a63022111f159d32.jpg  
      inflating: ./clothes_dataset/red_dress/c2467fb901400ecd3190db9089e6b3afd9016471.jpg  
      inflating: ./clothes_dataset/red_dress/c3396796cfb30f2391d98c1c77fe67e556020f42.jpg  
      inflating: ./clothes_dataset/red_dress/c398266b5e64ff74396ce033704212bc7b9eb488.jpg  
      inflating: ./clothes_dataset/red_dress/c46c542a4c4db337b023d2c4045e59512bb5e157.jpg  
      inflating: ./clothes_dataset/red_dress/c555e4d7da674e56d743086c0715236e25744b6a.jpg  
      inflating: ./clothes_dataset/red_dress/c55aede55313eb67494396099296b7f69cdcf3a7.jpg  
      inflating: ./clothes_dataset/red_dress/c59a228e55720641451f24091da4614dbdcd0370.jpg  
      inflating: ./clothes_dataset/red_dress/c59d399175786115d64b142f9f572a76ca2f92a4.jpg  
      inflating: ./clothes_dataset/red_dress/c59eddde645289db57a2b2730fecbcda09b45812.jpg  
      inflating: ./clothes_dataset/red_dress/c5b3532d5be359549fcdecc6fec39085eb6158fc.jpg  
      inflating: ./clothes_dataset/red_dress/c5d2160004c38593d99f4b04fa1db044e8e1ed3d.jpg  
      inflating: ./clothes_dataset/red_dress/c5ed922245175650aeef7903e747668f854080b3.jpg  
      inflating: ./clothes_dataset/red_dress/c63d984a073288c7ddc608ea585c288ec07118a7.jpg  
      inflating: ./clothes_dataset/red_dress/c673d42c612b55a89c0729444ea2b2b9bb022ab4.jpg  
      inflating: ./clothes_dataset/red_dress/c6dc5621c4d1da85a5c2efd82b36a684a3b29bcf.jpg  
      inflating: ./clothes_dataset/red_dress/c7519450ab03042f6b7394c63914876ac44e47e2.jpg  
      inflating: ./clothes_dataset/red_dress/c8267272a9dc0743de8801b6e1960c7b65122b11.jpg  
      inflating: ./clothes_dataset/red_dress/c879faef9a41f904580ace1a7ceff81cb9c996f9.jpg  
      inflating: ./clothes_dataset/red_dress/c9529e62fb8951b8a722d56c8910f1d3e2c3b6e6.jpg  
      inflating: ./clothes_dataset/red_dress/c95c509ed7059ae5a5ed284e533c1e48264783bb.jpg  
      inflating: ./clothes_dataset/red_dress/c9c1def859adbaef9f28d3b5eca775fc8578a12f.jpg  
      inflating: ./clothes_dataset/red_dress/c9e500783949427135af4651d7e0fa67ff90d136.jpg  
      inflating: ./clothes_dataset/red_dress/ca28ba0cb0d15783c1748bf04cb3db4509401eff.jpg  
      inflating: ./clothes_dataset/red_dress/ca2e7b531add723792d7baac82c4cd53cdf6fc0d.jpg  
      inflating: ./clothes_dataset/red_dress/ca927e76312db4f08fde3a0f30c96a89a09fa661.jpg  
      inflating: ./clothes_dataset/red_dress/cab110cda10396815b3a6eb8686e9cc221d53221.jpg  
      inflating: ./clothes_dataset/red_dress/cad7bbb2fac1db32e038cd26e637647e20ef358a.jpg  
      inflating: ./clothes_dataset/red_dress/caeac9b377fef9baa288f44b857884b7c7dc8acd.jpg  
      inflating: ./clothes_dataset/red_dress/cb52a2b383d2d26a54c69a35d0e348b6a7aeaa49.jpg  
      inflating: ./clothes_dataset/red_dress/cb65af616e80311f3843192c5b09b768718f9465.jpg  
      inflating: ./clothes_dataset/red_dress/cbb765d08ee98abb77ef0b64b02ec4bf581289ff.jpg  
      inflating: ./clothes_dataset/red_dress/cc5ff3c749b3454366ae587b3b9081a7b79e1545.jpg  
      inflating: ./clothes_dataset/red_dress/ccc57ba859c86c2ea806b1dde5039d2d4cd92946.jpg  
      inflating: ./clothes_dataset/red_dress/ccdd85c34f0a12be40b03eee95ffdd9a510cdf59.jpg  
      inflating: ./clothes_dataset/red_dress/ccfa58c9aa5a463791cb20216fd60412716a7eef.jpg  
      inflating: ./clothes_dataset/red_dress/cd0649ab2e9acebab80d664cda13ed3b5179c491.jpg  
      inflating: ./clothes_dataset/red_dress/cd521a730d22e12c26fc37c047514bd6e6404234.jpg  
      inflating: ./clothes_dataset/red_dress/cdb7e640fe7d704cd3712327d1d716832bf5dd66.jpg  
      inflating: ./clothes_dataset/red_dress/ce50ca45eef51e382cd5700a4f97183682f91402.jpg  
      inflating: ./clothes_dataset/red_dress/cef23c11980339355ad4274f26f45b8af27de77d.jpg  
      inflating: ./clothes_dataset/red_dress/cf0982f3bd4d2860a27a0caacf35b775cb2a0ac2.jpg  
      inflating: ./clothes_dataset/red_dress/cf1a2557e0e086a9606ec039c6da31fd4831d6ea.jpg  
      inflating: ./clothes_dataset/red_dress/d025ede11f2c35de482573cae88291430db6fbec.jpg  
      inflating: ./clothes_dataset/red_dress/d0297f98297243aaf639615425c544016ab260c4.jpg  
      inflating: ./clothes_dataset/red_dress/d0dada2697629ae95cb88142bb82e9776724d5f3.jpg  
      inflating: ./clothes_dataset/red_dress/d133c4fcfa6efda5be4ff399d9533d5ec7bd8c31.jpg  
      inflating: ./clothes_dataset/red_dress/d1c5327d1cfbc569e9f5ce384ac5de579de25c34.jpg  
      inflating: ./clothes_dataset/red_dress/d1ec8065951be47fbef90ce9e0beddc113ae7e59.jpg  
      inflating: ./clothes_dataset/red_dress/d1f0c1275b4f2940129c30e5f77a861673cbb87c.jpg  
      inflating: ./clothes_dataset/red_dress/d2715c8039d41d9af02cdb3a1b62f9a6ec2d9b7a.jpg  
      inflating: ./clothes_dataset/red_dress/d285399b23434c632bda91adc21a2de751fabe0c.jpg  
      inflating: ./clothes_dataset/red_dress/d37baa14a4183d6232850f695d66dbace3447718.jpg  
      inflating: ./clothes_dataset/red_dress/d381c9bb7ec726780ae56bfb55a5c967080174ca.jpg  
      inflating: ./clothes_dataset/red_dress/d3df59e528d3b91eab360abedadbfda28ddb1fc9.jpg  
      inflating: ./clothes_dataset/red_dress/d3e9ba42d0ca731fbbe2336b466832582d35d823.jpg  
      inflating: ./clothes_dataset/red_dress/d3f7b6b1fc7a12cc3a382f7220575f717023f7fd.jpg  
      inflating: ./clothes_dataset/red_dress/d4006b38dbc6cf5e4f5dc650c2edb27985892292.jpg  
      inflating: ./clothes_dataset/red_dress/d43a47b4dd4a8e7293a34cd60bf82c32bfe26f01.jpg  
      inflating: ./clothes_dataset/red_dress/d46069bf0d2c1cc6bbbd8463ca8d2d481ecd6f50.jpg  
      inflating: ./clothes_dataset/red_dress/d486db8e1361e8a200d1346f47c8abfb41eeeb62.jpg  
      inflating: ./clothes_dataset/red_dress/d49c4f76079656349bf33b47f50134478e3ad12c.jpg  
      inflating: ./clothes_dataset/red_dress/d5169b07c001ba14f34d9096301ff21def0374a6.jpg  
      inflating: ./clothes_dataset/red_dress/d5718bedab986c09a1de9bb1b21be7adb03ccecb.jpg  
      inflating: ./clothes_dataset/red_dress/d5977c0a771cf19f76b2ce8cce8700e0f3e48499.jpg  
      inflating: ./clothes_dataset/red_dress/d59a8e35d42b4988fda2d9f0bddd9a336ee1b91b.jpg  
      inflating: ./clothes_dataset/red_dress/d606556de2e82a3693cf24a5f9bccf7b62a879b0.jpg  
      inflating: ./clothes_dataset/red_dress/d66c96bf18736f6305f1650a4c7c008549220586.jpg  
      inflating: ./clothes_dataset/red_dress/d6d59f0234f6907439ca0acad548195b38f5fe9c.jpg  
      inflating: ./clothes_dataset/red_dress/d7e330d8349079158c246e72bc9da6b57904769b.jpg  
      inflating: ./clothes_dataset/red_dress/d82fc19535b92b9fef68ef6da13483897ccbeca1.jpg  
      inflating: ./clothes_dataset/red_dress/d8640a9c6f41e164cd524fee7b49049e013bf1ee.jpg  
      inflating: ./clothes_dataset/red_dress/d867eeb1c6369e20534580ed9404c01ee58b2bf7.jpg  
      inflating: ./clothes_dataset/red_dress/d912dc98ed0b3aeb1f0e24f35c9b4bbf287de14b.jpg  
      inflating: ./clothes_dataset/red_dress/d9321fa35400185f46709db65fd87cebf98bc338.jpg  
      inflating: ./clothes_dataset/red_dress/d9a85cb4226c931af8ed8c6da5355ad713e7d4ce.jpg  
      inflating: ./clothes_dataset/red_dress/da3b0618bdc603ed63aaa37679839e29caafb5ad.jpg  
      inflating: ./clothes_dataset/red_dress/db0a6d002d80c28ee0a2d53e5d74e1d9776bc6be.jpg  
      inflating: ./clothes_dataset/red_dress/db1d629402f63e35d4f36f2237524864adda8b93.jpg  
      inflating: ./clothes_dataset/red_dress/db36188f54ba91c9238a340889eaed88cc1841fa.jpg  
      inflating: ./clothes_dataset/red_dress/db7d665c20dd64a4548c3a9ff2b28d62121f66c5.jpg  
      inflating: ./clothes_dataset/red_dress/db904653bb24fc73aeea4d8bd6f654e7a2cb1f87.jpg  
      inflating: ./clothes_dataset/red_dress/dc2e37d79f0503147dd8e25c04a03d116e219958.jpg  
      inflating: ./clothes_dataset/red_dress/dc7a959914b0bc6605ba3555966be46aa0f595fd.jpg  
      inflating: ./clothes_dataset/red_dress/dcc24ded77e109ea114d261e68f3eba6923fce77.jpg  
      inflating: ./clothes_dataset/red_dress/dcd6da24ff8875027f65d122345d2d5493f83f82.jpg  
      inflating: ./clothes_dataset/red_dress/dd682cf4bb8ba2cdac44d58da7036e5c36cdcd53.jpg  
      inflating: ./clothes_dataset/red_dress/dd799dc4b0109142cc789b15bc78d0c7d566ebc7.jpg  
      inflating: ./clothes_dataset/red_dress/dd7c2b22385ac1d6e76bd3929172c9170e940b20.jpg  
      inflating: ./clothes_dataset/red_dress/dd8784204c6e19ef7197d73831353073f81a1c86.jpg  
      inflating: ./clothes_dataset/red_dress/de29ff7304b7b7fa608399a4a7e4171a03d34ab7.jpg  
      inflating: ./clothes_dataset/red_dress/de51c8dbbd684daa74ccee1501f02675875b4ad2.jpg  
      inflating: ./clothes_dataset/red_dress/de848197eeec1a0893aa37c81f061784df2fa0bd.jpg  
      inflating: ./clothes_dataset/red_dress/de8d8b38a14b462c2991bb534dfe4f876be8201d.jpg  
      inflating: ./clothes_dataset/red_dress/de9778e445f8d03c4f7ade68e88fca45aca2ab6f.jpg  
      inflating: ./clothes_dataset/red_dress/ded42f77853c6f74ea3df8987dbfb6be267d4628.jpg  
      inflating: ./clothes_dataset/red_dress/df19aad5aa95b98cf324cc96c4197a28db0943da.jpg  
      inflating: ./clothes_dataset/red_dress/df9164910984aa1fad9c68cadccf547372fda5b1.jpg  
      inflating: ./clothes_dataset/red_dress/dfacf7cb3b1d212f10ea6697d948da5cb9278b31.jpg  
      inflating: ./clothes_dataset/red_dress/e05ebf27d8eee8245d26a3678c7630301e255e7d.jpg  
      inflating: ./clothes_dataset/red_dress/e070ccfd27d0ab81a78d112a3a791b9ed24d570d.jpg  
      inflating: ./clothes_dataset/red_dress/e110c1bff420f998d63c8024eaf92502676878bf.jpg  
      inflating: ./clothes_dataset/red_dress/e12603ec3ab1484ea22d2466058f27e2bde2d76c.jpg  
      inflating: ./clothes_dataset/red_dress/e13ff9cd0932b9c1647c246d25aa48b4a53a0325.jpg  
      inflating: ./clothes_dataset/red_dress/e226e211f75cd431f132bf4ea38dff16ec036f2b.jpg  
      inflating: ./clothes_dataset/red_dress/e2aa3c2c035af5478d5353c653839099354f88e6.jpg  
      inflating: ./clothes_dataset/red_dress/e2c24aa5307b688e257022a69abf12fc5af66255.jpg  
      inflating: ./clothes_dataset/red_dress/e2d703677375c3e44e5fe33716b885ffc0432081.jpg  
      inflating: ./clothes_dataset/red_dress/e2e5c644d64c5b24baad14f0bfdfc55e81269a07.jpg  
      inflating: ./clothes_dataset/red_dress/e331728db2335df6dad28a9dbea915340459fdc8.jpg  
      inflating: ./clothes_dataset/red_dress/e34410f8ee415232e7a37e35b6abd8784e707694.jpg  
      inflating: ./clothes_dataset/red_dress/e34a76efe3fc17743af543e3215d2168c3a257e6.jpg  
      inflating: ./clothes_dataset/red_dress/e378efb7256e06173e79857f8eb438cff3c7dd4a.jpg  
      inflating: ./clothes_dataset/red_dress/e37f0f4b3e43ed4f9f3e8408d545c9d03fd4b105.jpg  
      inflating: ./clothes_dataset/red_dress/e417e3f525080c2a54ebf0b35e56dea74d99c9eb.jpg  
      inflating: ./clothes_dataset/red_dress/e42589d3678ac86021f7b8c8d4413f99daec46f2.jpg  
      inflating: ./clothes_dataset/red_dress/e433136236c3e958fb73f511d01a15caebc0f519.jpg  
      inflating: ./clothes_dataset/red_dress/e4b01729cbe1e2ed9daa6d621e3431c3aa88522d.jpg  
      inflating: ./clothes_dataset/red_dress/e582fe2bf664ae57edcd49c20f874a04b522d7c3.jpg  
      inflating: ./clothes_dataset/red_dress/e59d861e46f1939cdeff7ffb3e22f0256d958193.jpg  
      inflating: ./clothes_dataset/red_dress/e5ae8348171921f3deb00a3079084b7514d1d89a.jpg  
      inflating: ./clothes_dataset/red_dress/e60575894a8a08eeb9e5c15cf459bf209a32a915.jpg  
      inflating: ./clothes_dataset/red_dress/e6e8d5327ced88b648499cb63a41b74cff0a3e88.jpg  
      inflating: ./clothes_dataset/red_dress/e76515fd766cca52932e123f00230818727ac0c0.jpg  
      inflating: ./clothes_dataset/red_dress/e88716e9806bee53ece4cf2e48825d2d7df3bcff.jpg  
      inflating: ./clothes_dataset/red_dress/e8c39edd9ccf013ae9d4c60c81b449db0fb05a38.jpg  
      inflating: ./clothes_dataset/red_dress/e8dccf9164db4c5469b563a6a0fb3b995ca2459e.jpg  
      inflating: ./clothes_dataset/red_dress/e8ed786badaff8d80bb76166d0d3adec8c4550bd.jpg  
      inflating: ./clothes_dataset/red_dress/e97b3dd2e6328de50aae01aa90f18966e78269a6.jpg  
      inflating: ./clothes_dataset/red_dress/e995e03ca4033fb4c914fc4e95f7bf67e41e40dc.jpg  
      inflating: ./clothes_dataset/red_dress/e9de10a508f89ed281789847bdd33028ede45320.jpg  
      inflating: ./clothes_dataset/red_dress/ea025f37922e6e50b2f3170b6a9b4603a23ee1ea.jpg  
      inflating: ./clothes_dataset/red_dress/ea49182cd07f2abc7e7ad413ec0c40b2a841a7cc.jpg  
      inflating: ./clothes_dataset/red_dress/ea8c3de98a17e244dabf94954cbe0c62811bdbdf.jpg  
      inflating: ./clothes_dataset/red_dress/eb9efdf5e4d413389d2258cf29db471993f94c2e.jpg  
      inflating: ./clothes_dataset/red_dress/ebcb0aa92c2217c867f459917e39249697e5394f.jpg  
      inflating: ./clothes_dataset/red_dress/ebd2ea5d3682afa2b941059d9de2af8027a738cd.jpg  
      inflating: ./clothes_dataset/red_dress/ec53572105ed516e49c748bc4d614c6fef2780ce.jpg  
      inflating: ./clothes_dataset/red_dress/ec7335dbe2b03983898b26110261d85cce4a5ea6.jpg  
      inflating: ./clothes_dataset/red_dress/ecb4222be1786f0f142ad29e6f537368882b6e1b.jpg  
      inflating: ./clothes_dataset/red_dress/ed3eb2b35c74094d552839124089a999668ad8a2.jpg  
      inflating: ./clothes_dataset/red_dress/eed7ad3d8e1642f2ad37e2d6fb28cae86ae95d34.jpg  
      inflating: ./clothes_dataset/red_dress/eef4f750cb3d7e753010c5ed5f5611b33f7e1c84.jpg  
      inflating: ./clothes_dataset/red_dress/ef5ceda7bc555b9cc222f98596350d101307bcec.jpg  
      inflating: ./clothes_dataset/red_dress/ef9f0e9a55d53f790767bcbccd882b3fd229690a.jpg  
      inflating: ./clothes_dataset/red_dress/f01f7fcbb061fb8c3b6aff1572f1b261e3bd4dfb.jpg  
      inflating: ./clothes_dataset/red_dress/f1aa59809f9c27ed13f5571978f9ccbc60e29157.jpg  
      inflating: ./clothes_dataset/red_dress/f20d957e25a5d2e71a8d6862bf7d4c994346c755.jpg  
      inflating: ./clothes_dataset/red_dress/f22dcc27f7813982578969b00198936510a7cf9e.jpg  
      inflating: ./clothes_dataset/red_dress/f24cf0353f64a8905524b623c9ef822eabd7b1d3.jpg  
      inflating: ./clothes_dataset/red_dress/f28a9aed6cc0992c9499b9ef1bee145c143c7dec.jpg  
      inflating: ./clothes_dataset/red_dress/f34066253350a85871b6123ae98618fd2bb38cd3.jpg  
      inflating: ./clothes_dataset/red_dress/f3bb7981b8df52e276ee25db06bcb044ff6ff991.jpg  
      inflating: ./clothes_dataset/red_dress/f419c3b2d90202f5c6cc67f1842fa858cf925155.jpg  
      inflating: ./clothes_dataset/red_dress/f42504c4bc07ccde9a62cb949b0cc2c90bf96c32.jpg  
      inflating: ./clothes_dataset/red_dress/f45dfea2f24014173b5632ab55a24a155966a51b.jpg  
      inflating: ./clothes_dataset/red_dress/f47e15766046e4d8986c422ba37b6e74de1222da.jpg  
      inflating: ./clothes_dataset/red_dress/f4917c515b45210ed36da79e319d452293a90ee0.jpg  
      inflating: ./clothes_dataset/red_dress/f4b98cd02bcf4aff751da1649d1b4ae78531729b.jpg  
      inflating: ./clothes_dataset/red_dress/f4c508578974a4256abae370da6ca7c95eb3043f.jpg  
      inflating: ./clothes_dataset/red_dress/f4e6952655a7aa37612e8531d342cde0620c61d9.jpg  
      inflating: ./clothes_dataset/red_dress/f542c19aef3992774d2cbe5bb0a3f18990ad5d4e.jpg  
      inflating: ./clothes_dataset/red_dress/f58339d84849796337cd963fd3b4b1afbbcbfe82.jpg  
      inflating: ./clothes_dataset/red_dress/f5938c19f8a196a4952789c2b37fc7a645d81ab7.jpg  
      inflating: ./clothes_dataset/red_dress/f5fea225c4155957fcc33d1212d17fe42a3aed72.jpg  
      inflating: ./clothes_dataset/red_dress/f60d30171f122f2cc9579764130126925dcf3c18.jpg  
      inflating: ./clothes_dataset/red_dress/f65e2234311cf699dbac7a1525226149f5284491.jpg  
      inflating: ./clothes_dataset/red_dress/f6a1011e945d0fe5f5dfc6a6044e45c98c1ef876.jpg  
      inflating: ./clothes_dataset/red_dress/f6a6491232982d8a71b6452dde591dc82cb1bd04.jpg  
      inflating: ./clothes_dataset/red_dress/f6bd832d63b929337d817da9edd7db95b55d1475.jpg  
      inflating: ./clothes_dataset/red_dress/f911769b04473d8e756561aeea262c7b11c84d88.jpg  
      inflating: ./clothes_dataset/red_dress/f95a8b3313b64784da582f574b91ac877fee9f0d.jpg  
      inflating: ./clothes_dataset/red_dress/f9a18f1b4b509b0d56af165f1b488ea181cc8955.jpg  
      inflating: ./clothes_dataset/red_dress/f9daf5b3ca8b426f035bb839b940d9a5306f92bd.jpg  
      inflating: ./clothes_dataset/red_dress/fa6f74d67f9295b8b858102e251d065cbf248c12.jpg  
      inflating: ./clothes_dataset/red_dress/fb57ff2ff3b4619431d8b80ace66aa0225ce1096.jpg  
      inflating: ./clothes_dataset/red_dress/fb7ab73646e22b9f9d838431fdc095d13954ae24.jpg  
      inflating: ./clothes_dataset/red_dress/fc6aa5a502ed377d487bf7ec4ff3324dbba03b16.jpg  
      inflating: ./clothes_dataset/red_dress/fd509c3db042ad3b3112b87a996688d1dcb28f67.jpg  
      inflating: ./clothes_dataset/red_dress/fd59d8d5332690ed70a7504f2459de8db8b2789b.jpg  
      inflating: ./clothes_dataset/red_dress/fd7806187298129b1a5933e1ff5816013d5c2b5d.jpg  
      inflating: ./clothes_dataset/red_dress/fe416fca902e84aac1040d08164c7864d1be0ee0.jpg  
      inflating: ./clothes_dataset/red_dress/fea7f7ad5d7835b23a9c438ea0614192725c8729.jpg  
      inflating: ./clothes_dataset/red_dress/fee26c7101fedd1a48dc6c2cc91bf938d42f8668.jpg  
      inflating: ./clothes_dataset/red_dress/ff31a9d349c641ae70b343f85b8fe40f264b435f.jpg  
      inflating: ./clothes_dataset/red_dress/ff438775b5d6d955c9c7fd75aff8ac83229b9d79.jpg  
      inflating: ./clothes_dataset/red_dress/ff7624d462add656c2e59b8d6044ca264d685a8a.jpg  
      inflating: ./clothes_dataset/red_dress/ff921b6a11ce89d2ffc84ca2f2f01f97b25fd53d.jpg  
      inflating: ./clothes_dataset/red_dress/ffa5d04c6dbfe626f5723878a555dfcd60ecda7e.jpg  
      inflating: ./clothes_dataset/red_dress/ffc2ca86e2e0b9cb971a54ba22f1776fd54d9a29.jpg  
      inflating: ./clothes_dataset/red_dress/ffd6594e517c081c9f675a5b584fa10f91a9ef11.jpg  
      inflating: ./clothes_dataset/red_pants/00dd0c80730900182efe3460d3dce5b4300de987.jpg  
      inflating: ./clothes_dataset/red_pants/01e6894f6995cd0d61547431dcf1302d98850135.jpg  
      inflating: ./clothes_dataset/red_pants/0275e050baedb2eac9a512d138e871746a36c2a8.jpg  
      inflating: ./clothes_dataset/red_pants/03660484b7b9282283f3988297af1acd250f78d4.jpg  
      inflating: ./clothes_dataset/red_pants/0384125688806df1072d70192bce86efde468a61.jpg  
      inflating: ./clothes_dataset/red_pants/0395e52de1cea48477ee4bc5f2c1bd8c3506b09e.jpg  
      inflating: ./clothes_dataset/red_pants/05f83814a6da9cb316d5843f3150f388c0a65a17.jpg  
      inflating: ./clothes_dataset/red_pants/068c8a71ed0625a8d49126dff8604e977a90a394.jpg  
      inflating: ./clothes_dataset/red_pants/071fad6540c50607715355e834cad6e1601b3a92.jpg  
      inflating: ./clothes_dataset/red_pants/07243691bfa38ffe40d01236c138218c6650142a.jpg  
      inflating: ./clothes_dataset/red_pants/07faf8e2156c2f0903136e6bd6e2b6edf5e9d6ae.jpg  
      inflating: ./clothes_dataset/red_pants/09909231b7d28baa10bc0b9771f1a8e7f04f9a31.jpg  
      inflating: ./clothes_dataset/red_pants/09de2ad1f2ea951a86996f9fc687226850083722.jpg  
      inflating: ./clothes_dataset/red_pants/0b6cc71cb51056da4203546e8db1446419fdfa69.jpg  
      inflating: ./clothes_dataset/red_pants/0c2baac4b497b29ead67c695342f7eb7cfb9d755.jpg  
      inflating: ./clothes_dataset/red_pants/0c36b0484af19f4ba0234712fc15f97f0c39317a.jpg  
      inflating: ./clothes_dataset/red_pants/0caa21b88db10696ff8f70613055461ccbeec4a2.jpg  
      inflating: ./clothes_dataset/red_pants/0d46f82883cd8f1f2f485e590b92a6a48d786197.jpg  
      inflating: ./clothes_dataset/red_pants/0d5c329c4f43e21e7c123b3b98f320e8862dbdfa.jpg  
      inflating: ./clothes_dataset/red_pants/0dad097c973f0a1846baac5fb8bb57f4809906c3.jpg  
      inflating: ./clothes_dataset/red_pants/0db67f21685d22179fe3cf95f6559003e9e866c7.jpg  
      inflating: ./clothes_dataset/red_pants/0ee32a67c6943dc62a41945c1121f196af45251f.jpg  
      inflating: ./clothes_dataset/red_pants/100c16370ea72fdf2f85727164071ff3637341a1.jpg  
      inflating: ./clothes_dataset/red_pants/114903e58e160673f229821b035309864354e89a.jpg  
      inflating: ./clothes_dataset/red_pants/11a613ee81da70e086b5964880854effa1073262.jpg  
      inflating: ./clothes_dataset/red_pants/11f22f8d4938c70c8486a9f322fe80988f915ec0.jpg  
      inflating: ./clothes_dataset/red_pants/124dfc384ae6099d4b7be2801668667f74438494.jpg  
      inflating: ./clothes_dataset/red_pants/126aacb8318014ddfaec34f7453f25f552c781f9.jpg  
      inflating: ./clothes_dataset/red_pants/132cdb165756106e5ac7f0ef3d90667f573a491b.jpg  
      inflating: ./clothes_dataset/red_pants/140bc3730f17a83179bd663699f4f778e07a73d4.jpg  
      inflating: ./clothes_dataset/red_pants/1506543731898d8317e88531def365459b6ee8b5.jpg  
      inflating: ./clothes_dataset/red_pants/15e5c8e66834fea7d5ec22cc4ada753ac770d1a4.jpg  
      inflating: ./clothes_dataset/red_pants/15ed58a110756bc55a8862bd1559008532b7eab6.jpg  
      inflating: ./clothes_dataset/red_pants/172cb41a38d88eefffd233e060e3f28ae05f8460.jpg  
      inflating: ./clothes_dataset/red_pants/17c9cb39a63a3ac8fe9a93d5593c90b59ba85d54.jpg  
      inflating: ./clothes_dataset/red_pants/18077e7ec4b68081292fb4d36686805848e6e51d.jpg  
      inflating: ./clothes_dataset/red_pants/185bb443887718dc95eb60df2e292c0f058365ed.jpg  
      inflating: ./clothes_dataset/red_pants/18cc12feb887bb3f1ffcc06094c27ce12f5ce9dd.jpg  
      inflating: ./clothes_dataset/red_pants/19631ceaf9834543b28bd4281f5b49ade0f7871e.jpg  
      inflating: ./clothes_dataset/red_pants/1bc5be48012a980544a281417c314124a82e163f.jpg  
      inflating: ./clothes_dataset/red_pants/1c7e29b3a6c77cd44a317f3c056ff8018d580590.jpg  
      inflating: ./clothes_dataset/red_pants/1cfca894f71a71aabcf5740a7eada8929d70f8a8.jpg  
      inflating: ./clothes_dataset/red_pants/1d520e134f137fb07a1fd82284c519807b3239ed.jpg  
      inflating: ./clothes_dataset/red_pants/1d677e79fd24124636ab7b91df3fb039ff1396ca.jpg  
      inflating: ./clothes_dataset/red_pants/1dcf560dbb268c8d9675eef83a8f4e80be209118.jpg  
      inflating: ./clothes_dataset/red_pants/1de68916ba2a75f69e23d0aaeda62e89dc93e908.jpg  
      inflating: ./clothes_dataset/red_pants/1e1e46e2b172edb6eb623676153ebee25bee2b78.jpg  
      inflating: ./clothes_dataset/red_pants/1e8f86306db7e8d98426a37fdfd4bd463c6697e4.jpg  
      inflating: ./clothes_dataset/red_pants/1ece6124b68da7df0343fdedf0eb217632a5d769.jpg  
      inflating: ./clothes_dataset/red_pants/1f8de5e3c1a777a3dc25b8b57578cbeb4fbbb7c2.jpg  
      inflating: ./clothes_dataset/red_pants/1fedbca5670f689fc72b2c129760797a9fbe939f.jpg  
      inflating: ./clothes_dataset/red_pants/20ffd12d8a2073e4b271fbb2d6f29fa25854fc4e.jpg  
      inflating: ./clothes_dataset/red_pants/21d90f2a0f26bf5a1cff25bc99f46999f7147d0b.jpg  
      inflating: ./clothes_dataset/red_pants/21dcd93a46e777e905eac6d8ade5175defaa7800.jpg  
      inflating: ./clothes_dataset/red_pants/22b70b1af3cf0ff39db253013fbdabacc848a679.jpg  
      inflating: ./clothes_dataset/red_pants/27b34c70fcdf117298d53bccb1d945a59584f08d.jpg  
      inflating: ./clothes_dataset/red_pants/27c34a619df3716ed32f32b51768a80cfe093b1d.jpg  
      inflating: ./clothes_dataset/red_pants/28273dea0b699fdf6ac420dd928e84d1b9907631.jpg  
      inflating: ./clothes_dataset/red_pants/2cd8a84bc15028f9673b131df8b54662cab4844f.jpg  
      inflating: ./clothes_dataset/red_pants/2d2e50ad7407600d10f1bb6bd162085859599585.jpg  
      inflating: ./clothes_dataset/red_pants/2d60b0eca3fa5859b2e64de1ca7dde915a6653ec.jpg  
      inflating: ./clothes_dataset/red_pants/2e0dbf286034b927456e51c0054c037ab2d25d99.jpg  
      inflating: ./clothes_dataset/red_pants/2f212dadec4d4e9815a5ad8401b3264a27cfd101.jpg  
      inflating: ./clothes_dataset/red_pants/2ff92e33cba85fae7684e3df683836f501c0e64e.jpg  
      inflating: ./clothes_dataset/red_pants/3009a23caa2e7662a85742f418d064949c105e01.jpg  
      inflating: ./clothes_dataset/red_pants/302df35eaacada1ee0dc7e718e64e2c1d4b507ca.jpg  
      inflating: ./clothes_dataset/red_pants/306dfa924ac9af7fc776c29b8d1bf3e137234b70.jpg  
      inflating: ./clothes_dataset/red_pants/31a7d1cadab7ceafcd20511f4b5803522bfbe0ed.jpg  
      inflating: ./clothes_dataset/red_pants/35dc1b2e433112d161b51a4c3393100ff330086b.jpg  
      inflating: ./clothes_dataset/red_pants/369fc748c01e6f2094d7beb9ff95f3a2a0461a24.jpg  
      inflating: ./clothes_dataset/red_pants/371d79761e2f1c123c971684353f4e7710cb8193.jpg  
      inflating: ./clothes_dataset/red_pants/3768c7d70dc806442abd34b3f5212a90a07caf8e.jpg  
      inflating: ./clothes_dataset/red_pants/38a6304df8ab442f20fdd8c63b32e119bda54f76.jpg  
      inflating: ./clothes_dataset/red_pants/3a1671700dcb55f169bb718b756ce916f75bd9cf.jpg  
      inflating: ./clothes_dataset/red_pants/3b6574ab733938bf5303e5657427be3edda56281.jpg  
      inflating: ./clothes_dataset/red_pants/3e863fa077ee6f742cafe3d636f008088b5eb144.jpg  
      inflating: ./clothes_dataset/red_pants/3fa9ffa70e36562111dad891993b6cb1e6067836.jpg  
      inflating: ./clothes_dataset/red_pants/3feeb6f9093a43f116fb38f4acc2f6c60f25a6ec.jpg  
      inflating: ./clothes_dataset/red_pants/3ff3e56ce8024d9db18732375907761d89146d88.jpg  
      inflating: ./clothes_dataset/red_pants/4044310eecd6c1b63bf29fee60e04f057a266d53.jpg  
      inflating: ./clothes_dataset/red_pants/41ba68f363af26659ffe6aa173751a15436c673b.jpg  
      inflating: ./clothes_dataset/red_pants/4258ff73c5841146e4d8fcf4848ec2b94233158d.jpg  
      inflating: ./clothes_dataset/red_pants/4528c58a25f9cff6ff2ea0ddbd2bf857f50eead7.jpg  
      inflating: ./clothes_dataset/red_pants/45b89189a6cfd6c0cccbd8a736939c7a4e029dbb.jpg  
      inflating: ./clothes_dataset/red_pants/4632c2bf9d4a61ae97178d44419d9b3072c6e4df.jpg  
      inflating: ./clothes_dataset/red_pants/46da15416ef08aa4a8ad911cb3418f450bef05c5.jpg  
      inflating: ./clothes_dataset/red_pants/4798da24b59ab4a14cc61e3579a8c567e61490d5.jpg  
      inflating: ./clothes_dataset/red_pants/47bc3ab6d01a9d24d465520f21084c373583cbf9.jpg  
      inflating: ./clothes_dataset/red_pants/48a3174bc3827176953d749563c4db8aa75ec550.jpg  
      inflating: ./clothes_dataset/red_pants/4a937eb7ecfb7f7807759f987f1b53347d7f1bf3.jpg  
      inflating: ./clothes_dataset/red_pants/4d8cff2188f823942b462afa1135326a87ef8b5e.jpg  
      inflating: ./clothes_dataset/red_pants/4e6f766552c872ffbab23a72a79f3bc2de1c3245.jpg  
      inflating: ./clothes_dataset/red_pants/4edb1f4e2e4f3ddb8c6b289343023a1a202b0544.jpg  
      inflating: ./clothes_dataset/red_pants/4f03d8685dd8fba3dab5b55a468964f92d973d78.jpg  
      inflating: ./clothes_dataset/red_pants/4fbc03d037fddb24e01be7ea8ff6bb4e71c49730.jpg  
      inflating: ./clothes_dataset/red_pants/50078e91e963c27c97fca023a88810e2faedf6e8.jpg  
      inflating: ./clothes_dataset/red_pants/50e6620bb211d664a2accbe2ed793ab0e2298c4d.jpg  
      inflating: ./clothes_dataset/red_pants/51cb1a73cf2ff8bbc0153ac9c99101cf8769716b.jpg  
      inflating: ./clothes_dataset/red_pants/53a91ecab928fb6a98050ee636e3024b849b57b9.jpg  
      inflating: ./clothes_dataset/red_pants/5514d142e75e678b5db6713f2e875e7f82a9c9c7.jpg  
      inflating: ./clothes_dataset/red_pants/56547e4b2813ce11f88b7dee3f0bc85365803c91.jpg  
      inflating: ./clothes_dataset/red_pants/566b8e860b047e42a88250cf56e6fdd2539f53bd.jpg  
      inflating: ./clothes_dataset/red_pants/5746ddb92aa3f238412fda3360c6a040293cc7a9.jpg  
      inflating: ./clothes_dataset/red_pants/584f778aece14f07c2f386037c495cd7c2414c6d.jpg  
      inflating: ./clothes_dataset/red_pants/58a23c9a1efab122e05afd1e4bdd7cd806ecea1a.jpg  
      inflating: ./clothes_dataset/red_pants/5989c21b1827a0f341d6d2e284e3b155e65bd7e6.jpg  
      inflating: ./clothes_dataset/red_pants/5aa928157fcbc4e4688621d38bbebc681503eb8f.jpg  
      inflating: ./clothes_dataset/red_pants/5ab02f1ee7b3ee1579822611c02c250b4d47ba7a.jpg  
      inflating: ./clothes_dataset/red_pants/5aedf22ab0a3831bcf68028289fd44bbe1f073bf.jpg  
      inflating: ./clothes_dataset/red_pants/5ba5d1f7691bf335ea624ae48015c730c433781d.jpg  
      inflating: ./clothes_dataset/red_pants/5bb9eaab6b013c5facab1b11984a4ca6210bcfe7.jpg  
      inflating: ./clothes_dataset/red_pants/5becd3b8e6ce5a849a2fa3b89a4812f5a6e85a4d.jpg  
      inflating: ./clothes_dataset/red_pants/5bef18a4b8e6e0d4931005770a7b1eed4fb51404.jpg  
      inflating: ./clothes_dataset/red_pants/5c73f4e465eac84afa8f1231259bcddb327e51d6.jpg  
      inflating: ./clothes_dataset/red_pants/5cbd665023d19ab03014d2ba795feb4fbb4236a6.jpg  
      inflating: ./clothes_dataset/red_pants/5d0d589f48c3a89bb365c7ad082096727d76e34b.jpg  
      inflating: ./clothes_dataset/red_pants/5d9f04766f4a52660024386e0bad4cac727f9b2b.jpg  
      inflating: ./clothes_dataset/red_pants/5df3b503a15b61975037b08c4755d82c7a20131c.jpg  
      inflating: ./clothes_dataset/red_pants/5e5e8fa4f7f1a411fff20c0532002e98c2473728.jpg  
      inflating: ./clothes_dataset/red_pants/5eb34618330e40a903a183fa2666330167796449.jpg  
      inflating: ./clothes_dataset/red_pants/5f2a2ea75400e7f0dff68d8435033383794f3a03.jpg  
      inflating: ./clothes_dataset/red_pants/6014c5c0bf7492f490e29465de64568d9ad373cc.jpg  
      inflating: ./clothes_dataset/red_pants/61327e765d7caacafda83b038cbec52602b36393.jpg  
      inflating: ./clothes_dataset/red_pants/61d17fe88303447373c8892e3982ccb48ca1a146.jpg  
      inflating: ./clothes_dataset/red_pants/629ecce1601a32cf224939b8a167e958eea7a83c.jpg  
      inflating: ./clothes_dataset/red_pants/62deb56ca8b1b3e385b5a76b74ba6b2bd63ba119.jpg  
      inflating: ./clothes_dataset/red_pants/63005e03aebaa3361064bd1cb89578aacd418152.jpg  
      inflating: ./clothes_dataset/red_pants/63a64e2a7f1378d2d6319367ae750d9a9c63ea1c.jpg  
      inflating: ./clothes_dataset/red_pants/66e0322c01296595986a02ed9acf97b112c545c8.jpg  
      inflating: ./clothes_dataset/red_pants/67b3946fb8aa3883462e6f67e8656c7ca681ba60.jpg  
      inflating: ./clothes_dataset/red_pants/681c0c756b17996df057a354f55363ccbc6dbbf7.jpg  
      inflating: ./clothes_dataset/red_pants/688aa023b645e146b3491cca938fd5fed747946e.jpg  
      inflating: ./clothes_dataset/red_pants/699f5a12eb877c72d5df31bcfdd6ac92d7bbd1b3.jpg  
      inflating: ./clothes_dataset/red_pants/69a88cc419f54553e710c0478d4e940a836452b1.jpg  
      inflating: ./clothes_dataset/red_pants/69ffe21540e05941a2560aa2bd8564e866a6a0d1.jpg  
      inflating: ./clothes_dataset/red_pants/6b11961b2fbdffd4da11003caa79f025e33abeaf.jpg  
      inflating: ./clothes_dataset/red_pants/6b121cce0d0118c651acfc7235eb45da0d28a515.jpg  
      inflating: ./clothes_dataset/red_pants/6b71b9d0d9e00e699572afbd030d684536005682.jpg  
      inflating: ./clothes_dataset/red_pants/6bf61164929b92fc0f020409c77b63eb2de1ae9d.jpg  
      inflating: ./clothes_dataset/red_pants/6c4b1b93da351e31482b42a66ccaa64cdaad8e0e.jpg  
      inflating: ./clothes_dataset/red_pants/6d386f44a58cd7c04e393646dfcfdc98b804aa68.jpg  
      inflating: ./clothes_dataset/red_pants/6d920c6f9d8747de3b9172379c815767ed7a36c8.jpg  
      inflating: ./clothes_dataset/red_pants/6dbe76692c704cc7e808b4d86e3e6253e096b58f.jpg  
      inflating: ./clothes_dataset/red_pants/6e3b4d7dfb55850371c501e03d6a621c7bf3f877.jpg  
      inflating: ./clothes_dataset/red_pants/6e61d6f46a3cf3c133d646aada5cf99b3d5db9bd.jpg  
      inflating: ./clothes_dataset/red_pants/7126999103c54665c046bd184ef413d79017e5d1.jpg  
      inflating: ./clothes_dataset/red_pants/719931160f4faf9a7b85a4d51a2b707dcd7caa84.jpg  
      inflating: ./clothes_dataset/red_pants/7326f56f2d4bbe4614ae030ab067e4191cb1008c.jpg  
      inflating: ./clothes_dataset/red_pants/733fb8f2bf1b7fa9d21b1b562342763b94157a8e.jpg  
      inflating: ./clothes_dataset/red_pants/7435a738ba46fda8d357976d57651fca38d1e86f.jpg  
      inflating: ./clothes_dataset/red_pants/75f1b628c2ccc10c969b39a7137858aa34af465a.jpg  
      inflating: ./clothes_dataset/red_pants/77f108e5bf0871a782ebaf9f96f5698e48000244.jpg  
      inflating: ./clothes_dataset/red_pants/7c2b6e70ec8001555b4822673800f4b5e11411ef.jpg  
      inflating: ./clothes_dataset/red_pants/7c50f170894dac97ede77a9785ef8eadd1a3e049.jpg  
      inflating: ./clothes_dataset/red_pants/7ca1501fce91e7021d432fe44d0167a78744e367.jpg  
      inflating: ./clothes_dataset/red_pants/7d2b1e43065974d2c715d020ef883a89e34d17c4.jpg  
      inflating: ./clothes_dataset/red_pants/7d55263cc845693951c206f5ac19d1e7e6391317.jpg  
      inflating: ./clothes_dataset/red_pants/7d8f11e6261fd2e65cb4c7c247a0f402aa4c052c.jpg  
      inflating: ./clothes_dataset/red_pants/7fff7a7f86e7367e843d97a07a156bb4b2662467.jpg  
      inflating: ./clothes_dataset/red_pants/81a8dd86d83a7048d5b585f82ff5aefcd912d6d9.jpg  
      inflating: ./clothes_dataset/red_pants/83b0b3798d3b36e52a07c44668e5ebd354f93a18.jpg  
      inflating: ./clothes_dataset/red_pants/8458895096d2a910341bb416d025150f9b14f90b.jpg  
      inflating: ./clothes_dataset/red_pants/8584a9421f92833364959225e3744e87d117ecd1.jpg  
      inflating: ./clothes_dataset/red_pants/88d90b32f901318af989fcb1fc12916a84eee767.jpg  
      inflating: ./clothes_dataset/red_pants/898cff20c8acc77170bb41446423b3c6e3481fd0.jpg  
      inflating: ./clothes_dataset/red_pants/8b5fedd99939514bd2d409cea7db51d2974e75b9.jpg  
      inflating: ./clothes_dataset/red_pants/8bf5eb2d1c9d22a3fa1af40613ec62866111aa59.jpg  
      inflating: ./clothes_dataset/red_pants/8c66a41a848b26510cf6183d80afa1061e17dfce.jpg  
      inflating: ./clothes_dataset/red_pants/8c90d069ce2bb2675b2cafc6aac4771268a8cbc9.jpg  
      inflating: ./clothes_dataset/red_pants/8d1c6c21e05a051e4f0a577c1826a0cd5bb04499.jpg  
      inflating: ./clothes_dataset/red_pants/8d5f88ef1d918db03e2b2027c347fb16b3ac1bb1.jpg  
      inflating: ./clothes_dataset/red_pants/8e38bb33b21d0035eaecd8781fb9165956ab8ebd.jpg  
      inflating: ./clothes_dataset/red_pants/8ee0b5b31d4fb128f2d517bfa227f598449eaf5f.jpg  
      inflating: ./clothes_dataset/red_pants/8f4804708cde258876e660a11ce14090eef4a2fd.jpg  
      inflating: ./clothes_dataset/red_pants/907fb86d321d2b7490c9109b570f4b56dab45b4a.jpg  
      inflating: ./clothes_dataset/red_pants/918a6470308aa0496a94bcce9c16ccb67cccabf6.jpg  
      inflating: ./clothes_dataset/red_pants/9384f1410975f90a32e3b8c6b7f8bdd38389fecb.jpg  
      inflating: ./clothes_dataset/red_pants/93e977c6819910443b16488064e7b917ba9142b2.jpg  
      inflating: ./clothes_dataset/red_pants/93ed6c6ccd00762f4368e4a3cfe365bde418ad2e.jpg  
      inflating: ./clothes_dataset/red_pants/9429c9f9bb69699f9cca6b74cb6d5ae0cf032f1e.jpg  
      inflating: ./clothes_dataset/red_pants/944a9b6fafe454ff2705cd897f69575b7f42cce3.jpg  
      inflating: ./clothes_dataset/red_pants/9588d628a54606f2ab8e65c4ac8f6e4aa0ee80ce.jpg  
      inflating: ./clothes_dataset/red_pants/95b3b30443ec86172afb25d63c19ce76180073c2.jpg  
      inflating: ./clothes_dataset/red_pants/962c25bcc93ca1afc4c24768f37cc0d819f0b8c1.jpg  
      inflating: ./clothes_dataset/red_pants/963fa031fe3f6eb3de9c9f7b025aed7fe1225467.jpg  
      inflating: ./clothes_dataset/red_pants/9759046006607e178db682db07fe2a9a929c9684.jpg  
      inflating: ./clothes_dataset/red_pants/975e03d40abd3f844cdf36b565dba327a10ca234.jpg  
      inflating: ./clothes_dataset/red_pants/98550aa706443b4d12ebe4e74f7da8cba97f11df.jpg  
      inflating: ./clothes_dataset/red_pants/98a1cbf7a93d7b58fafdb25d9fd04ad63873bc3c.jpg  
      inflating: ./clothes_dataset/red_pants/9a362f3a584ec95793a9d77a1232058399b974ae.jpg  
      inflating: ./clothes_dataset/red_pants/9b1dfb55a19891563f6f3033d766b6483f3af1f9.jpg  
      inflating: ./clothes_dataset/red_pants/9ba799ced16ef4d8b4103ed9a66ef473fddccc72.jpg  
      inflating: ./clothes_dataset/red_pants/9c4d505b00265921d73e5216374e3126f876f089.jpg  
      inflating: ./clothes_dataset/red_pants/9ed1023c498c469754c5f155439d54baaf9b5ba2.jpg  
      inflating: ./clothes_dataset/red_pants/9ee3fdb3ba0c381b20bad5691e80b6b6408bc000.jpg  
      inflating: ./clothes_dataset/red_pants/9f7327b1d7c75f08c2a0439cee28a21ac685f822.jpg  
      inflating: ./clothes_dataset/red_pants/9fe8fb2615caf16658685a08184e652274f648bf.jpg  
      inflating: ./clothes_dataset/red_pants/a01387d7033953aeac944a750873302b7501c44d.jpg  
      inflating: ./clothes_dataset/red_pants/a0e513e366c81ee90e05ab785c0480645a5a770f.jpg  
      inflating: ./clothes_dataset/red_pants/a24a930c1f5fc44bd34784f6e25364ed4c11fed8.jpg  
      inflating: ./clothes_dataset/red_pants/a4495a8d6d697f06e9759847a12d67c307ba5661.jpg  
      inflating: ./clothes_dataset/red_pants/a820ed4f783730593c36237bc5b76ec0bfdda5d4.jpg  
      inflating: ./clothes_dataset/red_pants/a8648ecb225b6d2530a7db83b73a199a3cbe1ea2.jpg  
      inflating: ./clothes_dataset/red_pants/a8de84a205192969cd39ca6c016fedbe9f6bfa93.jpg  
      inflating: ./clothes_dataset/red_pants/a8e885c01c0f06aefbed4b2fbb5cf021193a7df9.jpg  
      inflating: ./clothes_dataset/red_pants/a9928bf28aa80f5d9a7328a5959a95c49f52c4e3.jpg  
      inflating: ./clothes_dataset/red_pants/aa91f030cfb96f67135b8bd9fd8868b3c6971454.jpg  
      inflating: ./clothes_dataset/red_pants/abd6749ff3dd9c2ac606b1d5209b40f44f5a6078.jpg  
      inflating: ./clothes_dataset/red_pants/ac62efc245bdd612eae37d49e2557e64a976425d.jpg  
      inflating: ./clothes_dataset/red_pants/ac8a68affc880d4dd81ebb1673171a856e1883dd.jpg  
      inflating: ./clothes_dataset/red_pants/ae521bbfbe56d4fa63551393aae7229ef86a3e69.jpg  
      inflating: ./clothes_dataset/red_pants/af2fcf23dbe68587c276a61b4e51425092261abc.jpg  
      inflating: ./clothes_dataset/red_pants/b05af5ffb451f7803f18a92918bd94e7892b8ba8.jpg  
      inflating: ./clothes_dataset/red_pants/b19e80ea2e75f106c73382730ac785e978b6cfd8.jpg  
      inflating: ./clothes_dataset/red_pants/b24602492420d664c4fd47b75181c769228faad2.jpg  
      inflating: ./clothes_dataset/red_pants/b2b695280987c7eb55bed9c8d5982a463c5912a8.jpg  
      inflating: ./clothes_dataset/red_pants/b316b0038f9828dff44e557cab26049406df3f6b.jpg  
      inflating: ./clothes_dataset/red_pants/b5b21b3362f72659060a57e4220fe67900e4f2e3.jpg  
      inflating: ./clothes_dataset/red_pants/b75aced534e741e3880823d051d4db4f69c91563.jpg  
      inflating: ./clothes_dataset/red_pants/b9afc97cd87199db7eb11f12b3a32e7759399e7a.jpg  
      inflating: ./clothes_dataset/red_pants/baf76c144974ae2965aaef79e9712bf6539b5445.jpg  
      inflating: ./clothes_dataset/red_pants/bb3d8a3f108801eeb6f09fae40b51494eb6e32a8.jpg  
      inflating: ./clothes_dataset/red_pants/bb4f0d968a865731a80bacc31781000064ede3aa.jpg  
      inflating: ./clothes_dataset/red_pants/bba325fb71a83e277c731a7d8b591203b290c25a.jpg  
      inflating: ./clothes_dataset/red_pants/bc19fc27215dfa22a468f1fd2e7b05170727ae9d.jpg  
      inflating: ./clothes_dataset/red_pants/bc40c173f258ae5067cda37940b120e532f78420.jpg  
      inflating: ./clothes_dataset/red_pants/bdfad4be4a051f8388e1aefff2b86eed1f975bf7.jpg  
      inflating: ./clothes_dataset/red_pants/be4870c5829e98b5c4f02529188dc155487a2988.jpg  
      inflating: ./clothes_dataset/red_pants/be939563c886bed700f156ce2738b0b7b83db593.jpg  
      inflating: ./clothes_dataset/red_pants/bf36bd8776c7ce24a72ec881a871d6e1f49628b9.jpg  
      inflating: ./clothes_dataset/red_pants/bfa38ff0c740ae1c0f52887f11dbe8241122ba55.jpg  
      inflating: ./clothes_dataset/red_pants/bfbebf5cd12c1648004cfd39fd11a7ba94720433.jpg  
      inflating: ./clothes_dataset/red_pants/c0cde2eb5e602cebab203f412a7759d111714883.jpg  
      inflating: ./clothes_dataset/red_pants/c1f127c3c74ba2ed6446fe158d3c2ec0353dd209.jpg  
      inflating: ./clothes_dataset/red_pants/c290d603a40a1416db2b8fe01f2df2cda106eaf1.jpg  
      inflating: ./clothes_dataset/red_pants/c3b292c36ad71c21b190b3e39883f155c6f5616c.jpg  
      inflating: ./clothes_dataset/red_pants/c3d7b4bb9fcc2513fb042d18cea0352ada8bd895.jpg  
      inflating: ./clothes_dataset/red_pants/c3d816c5a7c5caff150460179e65e65fc502c14e.jpg  
      inflating: ./clothes_dataset/red_pants/c55afec1b67475960d9e598ff8523d0535d8e1de.jpg  
      inflating: ./clothes_dataset/red_pants/c5d1fce9347b0e310359664b0914bf6c0197c4ac.jpg  
      inflating: ./clothes_dataset/red_pants/c60b026e09977afc23cbeeb1a02691694c116af7.jpg  
      inflating: ./clothes_dataset/red_pants/c61b5cedc4d6a9777fdadc753f3173ab842c171f.jpg  
      inflating: ./clothes_dataset/red_pants/c6815e31f2abb632672a0c505865aee93a6e550a.jpg  
      inflating: ./clothes_dataset/red_pants/c7d64ab95ab1a3856b7ffaa4c6e343d7d9303092.jpg  
      inflating: ./clothes_dataset/red_pants/c9f954032988265255f388413ed6fe0d67d875c7.jpg  
      inflating: ./clothes_dataset/red_pants/cb59151ec507e30506d51802ab943cc8238219c1.jpg  
      inflating: ./clothes_dataset/red_pants/cba7ac063a2b043708ecb7dc9ce3659cce736f45.jpg  
      inflating: ./clothes_dataset/red_pants/cd7a34b80a338b0e44602a85fbaa1d04ef42e3b3.jpg  
      inflating: ./clothes_dataset/red_pants/cddcc742365a1e02c7fc73d8ab4db00feab68029.jpg  
      inflating: ./clothes_dataset/red_pants/cde768f4cffa3db6bb971fda96f16e5fb0cc523c.jpg  
      inflating: ./clothes_dataset/red_pants/cec0607a24f4d434718c768a52b5105df785c495.jpg  
      inflating: ./clothes_dataset/red_pants/cf1bfb5e5394ca300b7e71ddb623f77939df8f60.jpg  
      inflating: ./clothes_dataset/red_pants/cf1c2005b24a771d718e70e758fd52c7afb95f72.jpg  
      inflating: ./clothes_dataset/red_pants/d263cb52de19edcb42102942c98bb3f767fdfeb3.jpg  
      inflating: ./clothes_dataset/red_pants/d26d2cf9ddd7f8a029a4d3211a6611161cb14bbf.jpg  
      inflating: ./clothes_dataset/red_pants/d399c2255fcbbd52154400c8444ba7cfb336d622.jpg  
      inflating: ./clothes_dataset/red_pants/d4481eb330f4efaa27a3d8b1b5711ec57613413f.jpg  
      inflating: ./clothes_dataset/red_pants/d6f40436a9cffaea5e8afd9ff0d5c979948cb3ec.jpg  
      inflating: ./clothes_dataset/red_pants/d86de990e5777f626cebf23b98df54f3e0384462.jpg  
      inflating: ./clothes_dataset/red_pants/d9bc76438062eba404abeeac0bc5732bf220be1c.jpg  
      inflating: ./clothes_dataset/red_pants/d9e0badc375cc04db20cf2a0d8b21882eb8204e9.jpg  
      inflating: ./clothes_dataset/red_pants/db9db80983dba49b29de510052b7bf1cdb3ce69b.jpg  
      inflating: ./clothes_dataset/red_pants/dbc6d527931649d15c9327219c9fe1a0c648dd93.jpg  
      inflating: ./clothes_dataset/red_pants/dc88806d2b455f2b118d2cc9757e99c86d835495.jpg  
      inflating: ./clothes_dataset/red_pants/df1f1273ab057d9a3583d3f1a3c6cd6e177aba25.jpg  
      inflating: ./clothes_dataset/red_pants/df46acf877c42bb0c9c408cd861ea6e547cd7f91.jpg  
      inflating: ./clothes_dataset/red_pants/e090259c696442fccc9ea2a780e57894893fc989.jpg  
      inflating: ./clothes_dataset/red_pants/e116ed38fcab02695838e02c27947e83a626fe4d.jpg  
      inflating: ./clothes_dataset/red_pants/e1bebee5e0da3d91e89c7dec43f73f3a048cdcea.jpg  
      inflating: ./clothes_dataset/red_pants/e25ee0956e4e9a10546e00793ed651726b231b6f.jpg  
      inflating: ./clothes_dataset/red_pants/e2dde2005a6f6dcd29499319620860fec4a91ec4.jpg  
      inflating: ./clothes_dataset/red_pants/e37885c40d353afcd36633ca0037d19e62169b86.jpg  
      inflating: ./clothes_dataset/red_pants/e3d258b68747a7366e677fa9b4ccae4400e7d346.jpg  
      inflating: ./clothes_dataset/red_pants/e3d438d95cde8cf78f1719a80fd25761da402d57.jpg  
      inflating: ./clothes_dataset/red_pants/e401868c4cebd7ee8739bbdd327fb263fdd4be20.jpg  
      inflating: ./clothes_dataset/red_pants/e413090f048a9e2e4d2d3d317885f8f020bb6369.jpg  
      inflating: ./clothes_dataset/red_pants/e4a28a6b78515d2f21355d73ac117553b06bd6d4.jpg  
      inflating: ./clothes_dataset/red_pants/e4ba4c424f606cc2acb87b7a93acc1c13a7bff71.jpg  
      inflating: ./clothes_dataset/red_pants/e4e9933386f3a507934616799fd6205fc7bcb5e9.jpg  
      inflating: ./clothes_dataset/red_pants/e51273756fbe042c15e2527aba4a0dac1dd0f6ed.jpg  
      inflating: ./clothes_dataset/red_pants/e5760854dc7b73ad4f7cfba38c530367ee7a4219.jpg  
      inflating: ./clothes_dataset/red_pants/e6186cdbb65a61dffe41787c32ac9e4a21c43faa.jpg  
      inflating: ./clothes_dataset/red_pants/e72d1c90b069fd8b2c5a6aac014a1c442e21444f.jpg  
      inflating: ./clothes_dataset/red_pants/e7f6bf5f1e625c34154e4ecf6a532dc263287a41.jpg  
      inflating: ./clothes_dataset/red_pants/eb4bbf3e5717fbce7d1da19a158b4ec806061057.jpg  
      inflating: ./clothes_dataset/red_pants/ec04e0e47bf362cac254ce2029bc8b12d324d954.jpg  
      inflating: ./clothes_dataset/red_pants/ec698f86f1a6feb66dab783037e3bbc1131c85f4.jpg  
      inflating: ./clothes_dataset/red_pants/efc1eedf955d183dc55e68db149489687d1e7e83.jpg  
      inflating: ./clothes_dataset/red_pants/f039d2b7f2eb7309ebc2098baa79d1dabdc36f7e.jpg  
      inflating: ./clothes_dataset/red_pants/f10d633b5db0ac38086efb2ac1a9de23a4d648f3.jpg  
      inflating: ./clothes_dataset/red_pants/f30e91ca3d6f493d8cabd82f47a93a1f307bb18e.jpg  
      inflating: ./clothes_dataset/red_pants/f3497f9614f1db50bf3426865a1f661cba3f3180.jpg  
      inflating: ./clothes_dataset/red_pants/f433cca80f974cf87c2b4adfa4260fd38ed94ee6.jpg  
      inflating: ./clothes_dataset/red_pants/f4c63aa98ee96ae5d6cd58396096859a000ce46d.jpg  
      inflating: ./clothes_dataset/red_pants/f57b292e96a0b9c94970869c838ddb69f3143108.jpg  
      inflating: ./clothes_dataset/red_pants/f625c967865c3290132ee19d1deed14b95109f30.jpg  
      inflating: ./clothes_dataset/red_pants/f71d085c680e71b3b3460dc0d8f13a16f46e6f01.jpg  
      inflating: ./clothes_dataset/red_pants/f9e97473d724f5a272d8a8b708299965e80ad8ad.jpg  
      inflating: ./clothes_dataset/red_pants/fa2d8f66819380bded3c4fbd4fb43d3715a6afec.jpg  
      inflating: ./clothes_dataset/red_pants/fa7a31cc0b1756b92912e7d7d20a0fdb5fd1b3ef.jpg  
      inflating: ./clothes_dataset/red_pants/fbaf0efe06e072886b91b4884f97124912dc7f90.jpg  
      inflating: ./clothes_dataset/red_pants/fbb9356b6b968e1b0481732f38f914ec166688ec.jpg  
      inflating: ./clothes_dataset/red_pants/fd5c57ae01caf225d432147a47042dbfc95a0ce7.jpg  
      inflating: ./clothes_dataset/red_pants/fdff3d0e0c686ec2c2513a145fefcbbf23559b87.jpg  
      inflating: ./clothes_dataset/red_pants/fe0ab8cf586a9cf2380e7b13b7baf606f571f2ef.jpg  
      inflating: ./clothes_dataset/red_pants/fe39630ca29472fcaff74a41a83841149c54cdbc.jpg  
      inflating: ./clothes_dataset/red_pants/fe92ce10c1eb46d6c49ea0839f927099ee6c550b.jpg  
      inflating: ./clothes_dataset/red_pants/fedd6c2476d18f1180efa8a1a7625e55addefd7b.jpg  
      inflating: ./clothes_dataset/red_shoes/005c117149e73f3bd66e6443d7b89db510d26ef5.jpg  
      inflating: ./clothes_dataset/red_shoes/0096bb54b5d1b12b1977cc386543b0a8c5518c7b.jpg  
      inflating: ./clothes_dataset/red_shoes/00dffd54941817035dfd2068e4c483b8f9850938.jpg  
      inflating: ./clothes_dataset/red_shoes/00eed5d51ab73063693790494dbbd0faa504c0bf.jpg  
      inflating: ./clothes_dataset/red_shoes/0147389a85ba60a9f5f295e72c6d99508dc70676.jpg  
      inflating: ./clothes_dataset/red_shoes/0196e9e4968e00a45c6fd113a34889a1e5816ef4.jpg  
      inflating: ./clothes_dataset/red_shoes/01a3386d4b0f45bda2bc574cf9f116c3b7fd87d3.jpg  
      inflating: ./clothes_dataset/red_shoes/01cd8e307a708cbc1e87d228e3b37c445d19c7af.jpg  
      inflating: ./clothes_dataset/red_shoes/025adda5b6b51f2cc4aee0c791c8bc2200ec52cd.jpg  
      inflating: ./clothes_dataset/red_shoes/02edd667536417a2d9b4d27df76ea3e5140c09c0.jpg  
      inflating: ./clothes_dataset/red_shoes/02fc9e0f3e8f583132d9c0572321f27f3d45c0cf.jpg  
      inflating: ./clothes_dataset/red_shoes/033df682d86b565a54a590662d61cdc5073dfdbb.jpg  
      inflating: ./clothes_dataset/red_shoes/0342c5f7757f62abaa28483707808589ed891698.jpg  
      inflating: ./clothes_dataset/red_shoes/03db90100edd286664b90c83e435decba6c75ac2.jpg  
      inflating: ./clothes_dataset/red_shoes/060c8d3a3f5574e0c602c0e829c8212e505e8e96.jpg  
      inflating: ./clothes_dataset/red_shoes/063d06d328ffd2c51e86c35ff22f4b8bf9deeede.jpg  
      inflating: ./clothes_dataset/red_shoes/06cad7e09cd4979bb72a2f7dcf0cfa4a08b3edb4.jpg  
      inflating: ./clothes_dataset/red_shoes/0733a8525a9d98a458f3f13a1661ecbdf1ba0eb4.jpg  
      inflating: ./clothes_dataset/red_shoes/0745cb02225e34c2654e4b6bc1d47442e27acb7c.jpg  
      inflating: ./clothes_dataset/red_shoes/07c28a61859dc65de9e668306b0d40a0a3cea8ff.jpg  
      inflating: ./clothes_dataset/red_shoes/07dbb9cf81b7898f4dd397513b72dd711c76dd4a.jpg  
      inflating: ./clothes_dataset/red_shoes/08fd299b981deb741407de50ef91822b7bac2fd0.jpg  
      inflating: ./clothes_dataset/red_shoes/091987b089f0029c373d68fa6536d86f7f0ea5c1.jpg  
      inflating: ./clothes_dataset/red_shoes/096acede0da9ad1f31cb1370b17f3c423addbfa5.jpg  
      inflating: ./clothes_dataset/red_shoes/09723b37d24131a90220a6fad486088400349676.jpg  
      inflating: ./clothes_dataset/red_shoes/0990ffbcb5689afab1c506d341dde5d07e3e0f61.jpg  
      inflating: ./clothes_dataset/red_shoes/09a5e55393491036ddd655f5030ad44093dbcbe9.jpg  
      inflating: ./clothes_dataset/red_shoes/09e2910ec5bb85d3f3bd041cce7bc95033c7f620.jpg  
      inflating: ./clothes_dataset/red_shoes/0a02fca2ac5dfff55c7b25e2853097adfa0358bf.jpg  
      inflating: ./clothes_dataset/red_shoes/0a467f9f29da199d659bf463094c38defff7684c.jpg  
      inflating: ./clothes_dataset/red_shoes/0acb5179fdccb3705dda138acb2160a02e8f3b9e.jpg  
      inflating: ./clothes_dataset/red_shoes/0b106ac42660e987e58e8fc8655b980b8540edb0.jpg  
      inflating: ./clothes_dataset/red_shoes/0b3726d59e11d7164b525673b518fec605deb2c6.jpg  
      inflating: ./clothes_dataset/red_shoes/0c71e58f603ef2e5886d3e884bf6880dae321017.jpg  
      inflating: ./clothes_dataset/red_shoes/0c755790db1232d32eb5f5ad165075df75764e68.jpg  
      inflating: ./clothes_dataset/red_shoes/0cbc07f32f8dc17ebf60331cb7b21c73b26c0b1c.jpg  
      inflating: ./clothes_dataset/red_shoes/0cc00ffc82544f33af5af66f312f40953388b0cc.jpg  
      inflating: ./clothes_dataset/red_shoes/0d40de1ff38471f5a5ec384b41fe153665f21a86.jpg  
      inflating: ./clothes_dataset/red_shoes/0d5b007343ae7def839f969cb66d9ab1675cbd9c.jpg  
      inflating: ./clothes_dataset/red_shoes/0de571c38c8c3d090e0c3168f4f754b1f7f7fdda.jpg  
      inflating: ./clothes_dataset/red_shoes/0e12fb834f227024841b8f6f993ead6b07f4827d.jpg  
      inflating: ./clothes_dataset/red_shoes/0e24e6fd1eb297ce5639d2ec7d5ebce1372eec99.jpg  
      inflating: ./clothes_dataset/red_shoes/0ec2c73422c51d49f2b6dafbfb54037556a1b89f.jpg  
      inflating: ./clothes_dataset/red_shoes/0fb88a53b1228f69e63cc0047cb66f8110eebdd6.jpg  
      inflating: ./clothes_dataset/red_shoes/112495fa91582f467d245b537787fff2b89b6367.jpg  
      inflating: ./clothes_dataset/red_shoes/11e9e3cfc2cc464dd11958c33e1ff802252fe9c0.jpg  
      inflating: ./clothes_dataset/red_shoes/125292d70099b660ca6e44367c3be26c1a08d713.jpg  
      inflating: ./clothes_dataset/red_shoes/12679f54ec36d50955670ddac3104cefaa80d91a.jpg  
      inflating: ./clothes_dataset/red_shoes/12a67ccf4fa4d78d25aa3092ac69b2a456ccf499.jpg  
      inflating: ./clothes_dataset/red_shoes/1410180992c13ad55db3604073fea9ad1b24d9da.jpg  
      inflating: ./clothes_dataset/red_shoes/1417f60b0b3ae256e8e1e721ab0f15a9ed11080a.jpg  
      inflating: ./clothes_dataset/red_shoes/142bd94cd767dac57632747d279f4a101ad8a71c.jpg  
      inflating: ./clothes_dataset/red_shoes/1562ab85f4bb60554a954aa963bc71d436f13992.jpg  
      inflating: ./clothes_dataset/red_shoes/15aa273c95fc4c4d4dc3ee768d235a2d2e41a478.jpg  
      inflating: ./clothes_dataset/red_shoes/16e19fd730c4ab106e890e58b382544501de46ac.jpg  
      inflating: ./clothes_dataset/red_shoes/172491a6718181b1bd14a80378124b0c7766b688.jpg  
      inflating: ./clothes_dataset/red_shoes/17c6d3afd14636f5b1f69c369d31afb6ea861d48.jpg  
      inflating: ./clothes_dataset/red_shoes/17fd18f8b24e9590b6f2467f232f00a074d83d35.jpg  
      inflating: ./clothes_dataset/red_shoes/18304208d76d2214e3b45853ff9f72100ddada6a.jpg  
      inflating: ./clothes_dataset/red_shoes/18909aa99d6ad2f1add7028c222526c7025fb833.jpg  
      inflating: ./clothes_dataset/red_shoes/1a28d6a0b74b4b6446cb5ea866abd870d7c46db3.jpg  
      inflating: ./clothes_dataset/red_shoes/1a572c5a832513cb750da7cf5090c4feaf593b8a.jpg  
      inflating: ./clothes_dataset/red_shoes/1b16171dd8bacdb02a46a036f8fca5c156c5de09.jpg  
      inflating: ./clothes_dataset/red_shoes/1b87beb5f67643a2a9571d004ab9f280277ca2e0.jpg  
      inflating: ./clothes_dataset/red_shoes/1d2a4662137bfc414245ca890955e195cb1dcbf1.jpg  
      inflating: ./clothes_dataset/red_shoes/1d7c06d237ac6614a34fc4189ed4d2bf3ce93ba7.jpg  
      inflating: ./clothes_dataset/red_shoes/1e1af17b76651580d94810b275c5a2b700dbe571.jpg  
      inflating: ./clothes_dataset/red_shoes/1e3bef440f37bd5d530b333713f45dee082232bf.jpg  
      inflating: ./clothes_dataset/red_shoes/1eae411ad59c5da9c83f39a12fd3d8f2ef995dc5.jpg  
      inflating: ./clothes_dataset/red_shoes/1fb51c8c0aa02f08dbd76212537f0bed26db68ea.jpg  
      inflating: ./clothes_dataset/red_shoes/1fdfbd1910513a6395c0e6ab8a38232c90ca5476.jpg  
      inflating: ./clothes_dataset/red_shoes/2006f5545f6ac435703415df88715acf29ab42ac.jpg  
      inflating: ./clothes_dataset/red_shoes/2028923c123693ba894f25c2abcc74c35d192dc5.jpg  
      inflating: ./clothes_dataset/red_shoes/213659b8ebc85b59aa4cd214d2220e2468a7aa81.jpg  
      inflating: ./clothes_dataset/red_shoes/219cdc0b2d7eada67525da8e5fb3146e882f2f78.jpg  
      inflating: ./clothes_dataset/red_shoes/219dec3e82afa9e6d11473b1a41a08a9602e79e1.jpg  
      inflating: ./clothes_dataset/red_shoes/21e789df0606eecb09594f496f624ac145761552.jpg  
      inflating: ./clothes_dataset/red_shoes/223967c9a62aac28478f3c2827890aad148acdcc.jpg  
      inflating: ./clothes_dataset/red_shoes/227cfcf407db8c9ff97f5e8e0d7a5773bec0de9b.jpg  
      inflating: ./clothes_dataset/red_shoes/22835dbfd28f01b065b4aaec62a775c4c90e6c94.jpg  
      inflating: ./clothes_dataset/red_shoes/233af2a8b103dbf7d8cc669aa7f288a783371032.jpg  
      inflating: ./clothes_dataset/red_shoes/235ad3c8d78acddf964139113777bb2d99dcda56.jpg  
      inflating: ./clothes_dataset/red_shoes/23f7ab92774f9ceca6adf99898ff1e4f5a4bc1cc.jpg  
      inflating: ./clothes_dataset/red_shoes/2539f590b4a90b04bc6a7173b378506ce24697d2.jpg  
      inflating: ./clothes_dataset/red_shoes/25bbf3b6dd71611ca15bf5d65de36c38211f27dc.jpg  
      inflating: ./clothes_dataset/red_shoes/2651ee648f5980af56cc244f642bcb3dd231f65d.jpg  
      inflating: ./clothes_dataset/red_shoes/266cdb0d0dddcf8d2e83cf2fb885e2d77b937dd4.jpg  
      inflating: ./clothes_dataset/red_shoes/26cef7f6dac50086a11fd6e2be7289a2021ebb2b.jpg  
      inflating: ./clothes_dataset/red_shoes/271e20db4b63acb06e045b3b0dae40927bcb66e2.jpg  
      inflating: ./clothes_dataset/red_shoes/2819d85e24ed6319e8c9f5e1a749e223dc62140e.jpg  
      inflating: ./clothes_dataset/red_shoes/2829e07e63d24ac533b0a2da682c7fe8d88649ef.jpg  
      inflating: ./clothes_dataset/red_shoes/28894729e25c7fa6cc9a0fa1e74f4833dbe20f34.jpg  
      inflating: ./clothes_dataset/red_shoes/28fe72c4a580926101a8f70ef743ad5574a98c1b.jpg  
      inflating: ./clothes_dataset/red_shoes/293da8ed02fd0176add9b65d1905ff04e0d9883d.jpg  
      inflating: ./clothes_dataset/red_shoes/29556b43cf52b095194adac7d06289748fac43e7.jpg  
      inflating: ./clothes_dataset/red_shoes/296a7e540b7a76d07f473ad0f6dbbdfa37f2b0a2.jpg  
      inflating: ./clothes_dataset/red_shoes/29c07cb0502983944bcb3b881a08d580ce90b1fd.jpg  
      inflating: ./clothes_dataset/red_shoes/2b43668b6b35d17df249576f9817333d29df858a.jpg  
      inflating: ./clothes_dataset/red_shoes/2c6ee5ac7c3e81dee9c5b7b7ba1e84353a74c9a4.jpg  
      inflating: ./clothes_dataset/red_shoes/2c725a069a5c3e0b3b0121ec1106e6807191e058.jpg  
      inflating: ./clothes_dataset/red_shoes/2cdaa7aafa8a9605daaa375d9872afeea7b9087b.jpg  
      inflating: ./clothes_dataset/red_shoes/2ce75cfa5e8350612fd5d02b17f4447d6595d062.jpg  
      inflating: ./clothes_dataset/red_shoes/2e75795602375f8f5f351850cf7af2f124bc6d53.jpg  
      inflating: ./clothes_dataset/red_shoes/2eb4493b64eae78d5a53568fa72605fcbfd1073c.jpg  
      inflating: ./clothes_dataset/red_shoes/2f8755e2481b8dd6f5612e91ad9e4cdd3061626a.jpg  
      inflating: ./clothes_dataset/red_shoes/2feb0fba429fb0def0b545195df14f50bba6687d.jpg  
      inflating: ./clothes_dataset/red_shoes/2feb394129f4453fe0c89a5774213a5fca7b559f.jpg  
      inflating: ./clothes_dataset/red_shoes/304e274a050d6ca025b6bd851d47e868ae8221ad.jpg  
      inflating: ./clothes_dataset/red_shoes/307367d1226f2d907f4abe7deed4d5226b100d54.jpg  
      inflating: ./clothes_dataset/red_shoes/309b88624bffeeb0e93aac67c957aec69bd06e15.jpg  
      inflating: ./clothes_dataset/red_shoes/30a76d7a7ec8480954dfef14a29bc464ac7fc64d.jpg  
      inflating: ./clothes_dataset/red_shoes/316782803748484cfa16f5e2562e6e62e01392fb.jpg  
      inflating: ./clothes_dataset/red_shoes/322d3369c1651f98d96b964dd72a102571881c8c.jpg  
      inflating: ./clothes_dataset/red_shoes/323ab2254459d1f9875e319c5ff8db176841a7d2.jpg  
      inflating: ./clothes_dataset/red_shoes/3283d3c982fe84b66e7147d1c661b7ede0f7dfd1.jpg  
      inflating: ./clothes_dataset/red_shoes/33ba9e9c0833cbcb28d8db97123c11e0f75c9fdd.jpg  
      inflating: ./clothes_dataset/red_shoes/3463cea7d1c82ef326b96b4917d8cf0d70687b71.jpg  
      inflating: ./clothes_dataset/red_shoes/34fd93c71c6628e1702d9cb2f50a93d6b21349ee.jpg  
      inflating: ./clothes_dataset/red_shoes/3571466b768118e2109c99df699ab44b895a9052.jpg  
      inflating: ./clothes_dataset/red_shoes/357a29876f38e965f632b35bf56a6f929f0d0c38.jpg  
      inflating: ./clothes_dataset/red_shoes/358ff472ac1678369180fb94afd7d695113cae65.jpg  
      inflating: ./clothes_dataset/red_shoes/361e909225a360001c238f0c47539a83f20c4bb9.jpg  
      inflating: ./clothes_dataset/red_shoes/36e214255f9b5928253cec7c9965b40aa1db3813.jpg  
      inflating: ./clothes_dataset/red_shoes/37047c09af9b92af83b90d2ce8240c96a1f728d5.jpg  
      inflating: ./clothes_dataset/red_shoes/38fa70950dd826d839721e422eed9aff33aa74db.jpg  
      inflating: ./clothes_dataset/red_shoes/392972d4e55b6871ee712a7e0b98793441f2cfbc.jpg  
      inflating: ./clothes_dataset/red_shoes/396de1181b6e83f794724e39ed052bbbad5dc3cf.jpg  
      inflating: ./clothes_dataset/red_shoes/39b0607c46d57921e7c12b95810fc887dcdf8910.jpg  
      inflating: ./clothes_dataset/red_shoes/39c77f0b5fb9a97c89a93d5095e3e71d555c7b99.jpg  
      inflating: ./clothes_dataset/red_shoes/39d3055cd109cec94faa71e5d1050fd4f93d7b8b.jpg  
      inflating: ./clothes_dataset/red_shoes/39d60ef93e30035769a644b2b19530d15222c12c.jpg  
      inflating: ./clothes_dataset/red_shoes/3a90851b286c67aaf82a1c12f44d083d5758e702.jpg  
      inflating: ./clothes_dataset/red_shoes/3ab9dfb090fc3db3ccbe06d9c334303e43273a19.jpg  
      inflating: ./clothes_dataset/red_shoes/3aea23e32303e0ee899e33fc05c113342b499a35.jpg  
      inflating: ./clothes_dataset/red_shoes/3b09f8c70dcad2c504a2c96548c0f851367adb2d.jpg  
      inflating: ./clothes_dataset/red_shoes/3b326f29b9bef6999c1e83deb0544b9aa5028974.jpg  
      inflating: ./clothes_dataset/red_shoes/3bff1a20061e0007c7d8280c353617af02439134.jpg  
      inflating: ./clothes_dataset/red_shoes/3c67cb2e7993d40a70b3507f0caaf6c7445dddbd.jpg  
      inflating: ./clothes_dataset/red_shoes/3d3cafe14483995aa303a53e3f830e2baec32068.jpg  
      inflating: ./clothes_dataset/red_shoes/3f96d3e8b01cac5d93ba61ffe67d097b4d79021e.jpg  
      inflating: ./clothes_dataset/red_shoes/3fd3bd10b764f2e7effa6910685709774e97ae68.jpg  
      inflating: ./clothes_dataset/red_shoes/4011ad73a7a35940aa7037d9e6a1766371775ca0.jpg  
      inflating: ./clothes_dataset/red_shoes/4095c71202fe4fea098647ae94de5858412307ae.jpg  
      inflating: ./clothes_dataset/red_shoes/418458f10caa634a43995f8dab30f89be8aa6377.jpg  
      inflating: ./clothes_dataset/red_shoes/418a4fb04104cbf0b7e4e5d7bcc22098cf09a539.jpg  
      inflating: ./clothes_dataset/red_shoes/418d1623f14b631c11127830e9363836dfa5e3ae.jpg  
      inflating: ./clothes_dataset/red_shoes/41a4fe770e7a45f978e4e9dbdb433d0bf83a8962.jpg  
      inflating: ./clothes_dataset/red_shoes/41d3cecfac82a4685fc5ed84f9c50165295c1c8d.jpg  
      inflating: ./clothes_dataset/red_shoes/41dc4de40e59c7c5d0f693473248ebf0c2675ced.jpg  
      inflating: ./clothes_dataset/red_shoes/41ef189aa0ef93d9d1a27d877fa7fb2156b80632.jpg  
      inflating: ./clothes_dataset/red_shoes/42a470c75cc1eeb41cd3da21162b93d0e7ead54c.jpg  
      inflating: ./clothes_dataset/red_shoes/430844bcc65d35f86dec9391e3bcfeff32f09cca.jpg  
      inflating: ./clothes_dataset/red_shoes/4390e32ae77387ba230d7301daeb876613335790.jpg  
      inflating: ./clothes_dataset/red_shoes/43a4b1246f8cb6ddd7ab72914c06356c111c803a.jpg  
      inflating: ./clothes_dataset/red_shoes/43b638a07f7e57b19a8ffb45cde2a077d07e6947.jpg  
      inflating: ./clothes_dataset/red_shoes/43f2e53b0bb445ca9b699bd82868e9b3f50cd26c.jpg  
      inflating: ./clothes_dataset/red_shoes/444d8bc5da943144d237d714137ed56c157c26ec.jpg  
      inflating: ./clothes_dataset/red_shoes/4460675e3d1c0cc2fe8c2748f60cb992aa2dc0ac.jpg  
      inflating: ./clothes_dataset/red_shoes/447097dc209c020ebd001de3076163f522ff41d5.jpg  
      inflating: ./clothes_dataset/red_shoes/448272f94a878952b258db4e9b1970ff5a066b17.jpg  
      inflating: ./clothes_dataset/red_shoes/44893a04037fc86a7dc32ccf5daad59f0d8690c1.jpg  
      inflating: ./clothes_dataset/red_shoes/44b2d66a2a58d4c8e18d93d434c91623847932c7.jpg  
      inflating: ./clothes_dataset/red_shoes/44bf5d1f6a60e8d7c48729d6b1d36e6617789800.jpg  
      inflating: ./clothes_dataset/red_shoes/453001eb15d1b16c33494ddaa5b9797a6d72223e.jpg  
      inflating: ./clothes_dataset/red_shoes/45c4252b44e80ee23aa2a1d5f491204bc5f2b18b.jpg  
      inflating: ./clothes_dataset/red_shoes/45ea8e5cf58964674fb5ee8e687249c3abd34648.jpg  
      inflating: ./clothes_dataset/red_shoes/45ec875d5586f85d89e77ab82503e5659d3372fb.jpg  
      inflating: ./clothes_dataset/red_shoes/47b47e13bb1e0e095798e64b5f63fc71c1c9b4e6.jpg  
      inflating: ./clothes_dataset/red_shoes/4813fcdaad0c70db5fda686c0fdd3e44f25c7bbe.jpg  
      inflating: ./clothes_dataset/red_shoes/4957ca33d9ae9b6e84bb84180e715ebbecc76729.jpg  
      inflating: ./clothes_dataset/red_shoes/49e05db3c8ba54bfb3d95ce96a5efe4fa3d342dd.jpg  
      inflating: ./clothes_dataset/red_shoes/4a0ff0a278db8c805b697135d17247e5a6c170bc.jpg  
      inflating: ./clothes_dataset/red_shoes/4a310ab07da0aa878ba65eddc9d7e3ce8cd11455.jpg  
      inflating: ./clothes_dataset/red_shoes/4aa7822859f9cac2da0bf12f1a8e763f301c4eff.jpg  
      inflating: ./clothes_dataset/red_shoes/4ac67bd7fe29da5d6fa5478f3441eb08d23d6555.jpg  
      inflating: ./clothes_dataset/red_shoes/4ad5d5a73773c19b01e9f1c8817e81da5bc12d89.jpg  
      inflating: ./clothes_dataset/red_shoes/4ba1fb56cbb61c4b949406bf6fc4021d67596ed1.jpg  
      inflating: ./clothes_dataset/red_shoes/4bc28e539436cd3a00b32fc792f83ddb7ea02f11.jpg  
      inflating: ./clothes_dataset/red_shoes/4bebb926c5ac851c68ec16cca163f7aa6e20639c.jpg  
      inflating: ./clothes_dataset/red_shoes/4cc7a7f80d8aadaf45a1be64e9deced5d0740213.jpg  
      inflating: ./clothes_dataset/red_shoes/4d0e2727cb019fd3a1dffbced7be9c4b2f5ebc92.jpg  
      inflating: ./clothes_dataset/red_shoes/4d1312b61f1d396b090df51da954a6dcd04fde98.jpg  
      inflating: ./clothes_dataset/red_shoes/4ded49dd338d94057026d0c7a2ead1be7eed62dd.jpg  
      inflating: ./clothes_dataset/red_shoes/4ed37b7346b4b6dceb5b3cdc6cea863d2b743102.jpg  
      inflating: ./clothes_dataset/red_shoes/4f95378f3eb9caa5a172c9486dc22f6a3bb64327.jpg  
      inflating: ./clothes_dataset/red_shoes/4fbe1d4cc2401d7951da2c4e6e49ac5f4efc704b.jpg  
      inflating: ./clothes_dataset/red_shoes/5104ec0ad45ebd342f1eefef8fdbe823b067fbfc.jpg  
      inflating: ./clothes_dataset/red_shoes/5105fc335f198644ebc62387b0a8f2bac1bfc0b9.jpg  
      inflating: ./clothes_dataset/red_shoes/518e166d75ace6268070d2c39e5f934d6993c83a.jpg  
      inflating: ./clothes_dataset/red_shoes/5231ad779815b4e7dab0420a10985d3d345054b9.jpg  
      inflating: ./clothes_dataset/red_shoes/52eda8d7ee45ae6a91b9a3e123e7a2d3c5acbaf2.jpg  
      inflating: ./clothes_dataset/red_shoes/52ff42806e878febca202f5618fe09ff3ec5c16e.jpg  
      inflating: ./clothes_dataset/red_shoes/53576aa8908f149174c531fd468efc95f9e6a933.jpg  
      inflating: ./clothes_dataset/red_shoes/5365b7c5548e59cc4ef3f33eadfe6525453623db.jpg  
      inflating: ./clothes_dataset/red_shoes/53901ccba23d6200116d989b077d61705d8b229e.jpg  
      inflating: ./clothes_dataset/red_shoes/53ba16098730d569628d24a79d14142fe7195537.jpg  
      inflating: ./clothes_dataset/red_shoes/54441030e1b9cc08873a2e2a765ef8ae6f1bbc98.jpg  
      inflating: ./clothes_dataset/red_shoes/55ff5f291eecfebecc2d439e5d101421d6b906ad.jpg  
      inflating: ./clothes_dataset/red_shoes/5645e3c87b86e3fb81e905c14020e26429cf19f4.jpg  
      inflating: ./clothes_dataset/red_shoes/56e4b5bc6ca1ada4ed7a8bb37fe77ac5b6c11fea.jpg  
      inflating: ./clothes_dataset/red_shoes/57007b1e36f9b86f2832005bf20de8d3fe12b518.jpg  
      inflating: ./clothes_dataset/red_shoes/5795d17ce859663a31c0de73faba189bb9c874cd.jpg  
      inflating: ./clothes_dataset/red_shoes/58367a3cfa75c13b708db57d82a66a863be471f7.jpg  
      inflating: ./clothes_dataset/red_shoes/586ca417f3648a3a7c344e105a1bae2161e24743.jpg  
      inflating: ./clothes_dataset/red_shoes/5885ca548c824554b7423e083a841f1cbb3a10ee.jpg  
      inflating: ./clothes_dataset/red_shoes/599780cb22fbaea39a8f48ab0c0b6ad9b205f39d.jpg  
      inflating: ./clothes_dataset/red_shoes/59a14990ffa1110986d3ce105f32f55765513696.jpg  
      inflating: ./clothes_dataset/red_shoes/59c0df46e34ec72f4a8f33355408b7d722fe4e60.jpg  
      inflating: ./clothes_dataset/red_shoes/5a045eeba98ded738a0f220bfb52e15857bcb031.jpg  
      inflating: ./clothes_dataset/red_shoes/5a15a640a96e8c46c84dcaeaaf18ff708b553785.jpg  
      inflating: ./clothes_dataset/red_shoes/5b0081258ba4c583a0149fb42b484778613b4c24.jpg  
      inflating: ./clothes_dataset/red_shoes/5bb1811c949e7977d97fc0624ee4573603fcf910.jpg  
      inflating: ./clothes_dataset/red_shoes/5bd5d3d5c99f6c1f540c74bb5321da4604dd585d.jpg  
      inflating: ./clothes_dataset/red_shoes/5beef1adbb839cd623fd9cd49cec24cedb47994d.jpg  
      inflating: ./clothes_dataset/red_shoes/5c04690204e584ac346c44dde51eccb692ec970a.jpg  
      inflating: ./clothes_dataset/red_shoes/5c66cb26da5acbe927db2c761f2be46d6fa89811.jpg  
      inflating: ./clothes_dataset/red_shoes/5ca06c2b2c9f3b44989898c422b4d5e3c1167c5d.jpg  
      inflating: ./clothes_dataset/red_shoes/5cd117ccbb60e9625ae7c8aa9d61762978a8c439.jpg  
      inflating: ./clothes_dataset/red_shoes/5e006b1eab73efeaa91fb76aa7c2d6e24706e60f.jpg  
      inflating: ./clothes_dataset/red_shoes/5e5a2acf1da66fb678e54a1dbe67aaa405acf512.jpg  
      inflating: ./clothes_dataset/red_shoes/5fe29fd38d4c6da4537eb4805af481162839882b.jpg  
      inflating: ./clothes_dataset/red_shoes/609982179e0503e2f25a21abdb116757fe31ba51.jpg  
      inflating: ./clothes_dataset/red_shoes/60bf3d7008dd3c1b4da5d8d64524e8186afd944e.jpg  
      inflating: ./clothes_dataset/red_shoes/61adfd0adb5a5acd4493f6fe7aebfaab7e9b3b92.jpg  
      inflating: ./clothes_dataset/red_shoes/62730ae6115f9a06deadd0b8d90b7e620fc2d443.jpg  
      inflating: ./clothes_dataset/red_shoes/62d8102c9d47c8d354abeeb241411bf4690552bc.jpg  
      inflating: ./clothes_dataset/red_shoes/6359d0eabc5ccda30e133728564363b500e6ead3.jpg  
      inflating: ./clothes_dataset/red_shoes/63a106debb88ae89f77ae8c284cc8983213594ae.jpg  
      inflating: ./clothes_dataset/red_shoes/63a17b55e571ac4f6451c5a635c887ad4bb01c7b.jpg  
      inflating: ./clothes_dataset/red_shoes/64777047d5044b861ecb2c08aeb9e5965278ce88.jpg  
      inflating: ./clothes_dataset/red_shoes/654f5db494efecb8894e969f119aafa0768fe92d.jpg  
      inflating: ./clothes_dataset/red_shoes/65620f2ece0b567c10fc8d67d0507748b62026c5.jpg  
      inflating: ./clothes_dataset/red_shoes/65f0b7198c65b804ccc8b6c00d8dfcd1c3293fdf.jpg  
      inflating: ./clothes_dataset/red_shoes/660aa4f81d7fd082308ea644acd4a72e9b0e041f.jpg  
      inflating: ./clothes_dataset/red_shoes/66156fc5aa11c7852c11836560f4f2fda1b8686b.jpg  
      inflating: ./clothes_dataset/red_shoes/662102f3cd9a5973f7d65dca38b04a856af75eeb.jpg  
      inflating: ./clothes_dataset/red_shoes/666e1227da2e0bbd8170293934a26037e361fc6b.jpg  
      inflating: ./clothes_dataset/red_shoes/6674b4627462d80d556eb862db7e92ee6a010812.jpg  
      inflating: ./clothes_dataset/red_shoes/672c233b0f621e6f1167db0e315229cca74ed5f0.jpg  
      inflating: ./clothes_dataset/red_shoes/67d3b3a9ef6cc02343ca49fe52c6478e1b15ffc7.jpg  
      inflating: ./clothes_dataset/red_shoes/6818b5cc04c0cac53615afdc4a2d474f45994148.jpg  
      inflating: ./clothes_dataset/red_shoes/685eb653d720e352c6177f70e460284055729a37.jpg  
      inflating: ./clothes_dataset/red_shoes/68933a56d082539e905c4c70b34d9a557acb02a1.jpg  
      inflating: ./clothes_dataset/red_shoes/68e851540acc402944ca52f787e99f788dcfb290.jpg  
      inflating: ./clothes_dataset/red_shoes/6960b59a262122f5ff965864286fdad57ef1db6d.jpg  
      inflating: ./clothes_dataset/red_shoes/6963a008c43fce89dbd22023f30eb5de0068ec6a.jpg  
      inflating: ./clothes_dataset/red_shoes/69861b58cc675576fac38c8656f134ed86d89c21.jpg  
      inflating: ./clothes_dataset/red_shoes/69c10e7ee4dc3e8817fb910084bd442a61b20004.jpg  
      inflating: ./clothes_dataset/red_shoes/6add99775ee8e2c08500902a5dd8f112172f0241.jpg  
      inflating: ./clothes_dataset/red_shoes/6b7f29862cfd27895f964e76c8eb290dc8bd4e83.jpg  
      inflating: ./clothes_dataset/red_shoes/6c0a8271675544484a489ec1514d37d1dda7e971.jpg  
      inflating: ./clothes_dataset/red_shoes/6c11da19efa1c1e94f1152260e2081fecb1e65ad.jpg  
      inflating: ./clothes_dataset/red_shoes/6c14510df6b030048a60c8bdf46a0f09ab2496d5.jpg  
      inflating: ./clothes_dataset/red_shoes/6c473c74f9bfed1ba724344d1b4e3af37ebfbf63.jpg  
      inflating: ./clothes_dataset/red_shoes/6d566a28d6ea7748ca49805d1c9e82f5a6621a93.jpg  
      inflating: ./clothes_dataset/red_shoes/6d94cd6b8b7231a09a9fbea278f3eb0041801a95.jpg  
      inflating: ./clothes_dataset/red_shoes/6db5cda376e99b5201fc3ffbb410f0d259f25c4f.jpg  
      inflating: ./clothes_dataset/red_shoes/6dc442916758ee91fa85b2faa3862af68f8aa5f4.jpg  
      inflating: ./clothes_dataset/red_shoes/6e1a90801f6d3535afcfab09a291537975bbad1d.jpg  
      inflating: ./clothes_dataset/red_shoes/6e761b6a502a9e0b7f80ae66f63425ab63257140.jpg  
      inflating: ./clothes_dataset/red_shoes/6ef543f66b2b9ab2bc367dde890ef9b6be82a397.jpg  
      inflating: ./clothes_dataset/red_shoes/6f05157b657b37e4c28ae5a24ed26f7cb273fdc3.jpg  
      inflating: ./clothes_dataset/red_shoes/6f40d285a05e4818ef6e911f8657b22056a308aa.jpg  
      inflating: ./clothes_dataset/red_shoes/6f4d081ade59d444adc1aaafc81cb36a5e2412fb.jpg  
      inflating: ./clothes_dataset/red_shoes/6fbf217fa20086cb91b80721bfe1c0f8f6803f03.jpg  
      inflating: ./clothes_dataset/red_shoes/7036d637fa6249b93072f13230a3f3a25bc123a7.jpg  
      inflating: ./clothes_dataset/red_shoes/703a280f7fca4cbadbcf73518b991608ac0a87ed.jpg  
      inflating: ./clothes_dataset/red_shoes/704d89458dd65f2660a32b078cf2687b80924a75.jpg  
      inflating: ./clothes_dataset/red_shoes/7124e3d6a96c0cdecc1fdecde54eae4a746f5b22.jpg  
      inflating: ./clothes_dataset/red_shoes/71a51e916699b07e87da60a5662707e1c300c8e8.jpg  
      inflating: ./clothes_dataset/red_shoes/722750ae1ddc575b709cf27058c86c1fd217c2be.jpg  
      inflating: ./clothes_dataset/red_shoes/7284c0be5f8c267a8eb1c1c2b18648bb97657535.jpg  
      inflating: ./clothes_dataset/red_shoes/72e21536f069e1264e037821d2a640672ca81a5e.jpg  
      inflating: ./clothes_dataset/red_shoes/73585302461a54aedb0d547ab254e3a554d3d367.jpg  
      inflating: ./clothes_dataset/red_shoes/739661dc94d9686dd804f72b9338618dc0a80b75.jpg  
      inflating: ./clothes_dataset/red_shoes/744ee5b9bcc1c7d71b22ff82a3d247a36546b77a.jpg  
      inflating: ./clothes_dataset/red_shoes/74b2bdac9878a58f5c5094bcdd18ac1fca190727.jpg  
      inflating: ./clothes_dataset/red_shoes/75b74c775389d6a21e50a10865471cf349142b77.jpg  
      inflating: ./clothes_dataset/red_shoes/75ce91b66ea6692d1bf98506b8ed243812b272f1.jpg  
      inflating: ./clothes_dataset/red_shoes/761bca4aab1e77340dd4d3750f5c0e4cdb7c3c1e.jpg  
      inflating: ./clothes_dataset/red_shoes/76872c9b82300d10e7c30959ea95f52cdd82e495.jpg  
      inflating: ./clothes_dataset/red_shoes/76da4a2f9edad9b35493e01335118285d9efc29b.jpg  
      inflating: ./clothes_dataset/red_shoes/773a937c01d32a619be2db61e41930d167d964cf.jpg  
      inflating: ./clothes_dataset/red_shoes/778428f07a39109ef71f4cc2b4b95a552d91132b.jpg  
      inflating: ./clothes_dataset/red_shoes/779bf5da263a6421fed61aaaa7ba7bedc471bab4.jpg  
      inflating: ./clothes_dataset/red_shoes/780e3460bece36b968bc6f6cb045d23fbad053c6.jpg  
      inflating: ./clothes_dataset/red_shoes/7831381ab65f8114698e160d239414635d56318a.jpg  
      inflating: ./clothes_dataset/red_shoes/79dad714435af64c3031894b013424e60519d457.jpg  
      inflating: ./clothes_dataset/red_shoes/7a2446ec32670caef1625b57ceefb885743449ed.jpg  
      inflating: ./clothes_dataset/red_shoes/7b09059761517fec59183c5da3128b0e3e1d4942.jpg  
      inflating: ./clothes_dataset/red_shoes/7b89592cf9ece6f3f6b67925b1ce32178c7f4670.jpg  
      inflating: ./clothes_dataset/red_shoes/7c818b3d1ef0c27256b66e5f8dc31f8883033980.jpg  
      inflating: ./clothes_dataset/red_shoes/7c82ab64dd9a555fdb974853b5c27d0e495f0b18.jpg  
      inflating: ./clothes_dataset/red_shoes/7ce0591c7a0a59f6f79a7d01a4a6a5dec18f4b54.jpg  
      inflating: ./clothes_dataset/red_shoes/7d057e01c75a10756dc9023fbcfa858e38524fa1.jpg  
      inflating: ./clothes_dataset/red_shoes/7f0cdf880f63e9d787bff69258dca3ed8773f809.jpg  
      inflating: ./clothes_dataset/red_shoes/7f0e40a5fc8afd993b961a7bad96c0ea280c0790.jpg  
      inflating: ./clothes_dataset/red_shoes/7fc9a4b0bc22732e6c450b91b25d666116324619.jpg  
      inflating: ./clothes_dataset/red_shoes/802b317807e5253827412170b168fedcddc3348e.jpg  
      inflating: ./clothes_dataset/red_shoes/806dd0901d40f2ba9890a1e131f9309872b619a7.jpg  
      inflating: ./clothes_dataset/red_shoes/80fc15e9c0b40c589f312fb86888ac57d2e005ab.jpg  
      inflating: ./clothes_dataset/red_shoes/8201a67a6ce3af5f54fddfaca024edd3c0bfe083.jpg  
      inflating: ./clothes_dataset/red_shoes/8211e03c2117975576fe588c44fa11b5fed41ff6.jpg  
      inflating: ./clothes_dataset/red_shoes/826107e0737ea846b868caae90663514c74c0b7e.jpg  
      inflating: ./clothes_dataset/red_shoes/82cca56cb6ac983353141db66dcca26803759872.jpg  
      inflating: ./clothes_dataset/red_shoes/83418db970ca8487d97d15578bee55ee9c54ec11.jpg  
      inflating: ./clothes_dataset/red_shoes/839eb5d4d499818a8d60ef0f20545e7ba8206b1a.jpg  
      inflating: ./clothes_dataset/red_shoes/83acb9b5ca57a73fd9de20de5b2fcbdcb43b1ff4.jpg  
      inflating: ./clothes_dataset/red_shoes/83b4eb3a8cabea768b20cc48da290f4a925bd8da.jpg  
      inflating: ./clothes_dataset/red_shoes/84258d9b823d9bbdaeeca00d8fce1992f73019ce.jpg  
      inflating: ./clothes_dataset/red_shoes/84974aa150bacee09396da1e79b842d727b7eafa.jpg  
      inflating: ./clothes_dataset/red_shoes/84979cda837058e9a5daa3b87ca317796790a593.jpg  
      inflating: ./clothes_dataset/red_shoes/84db7a8bc88dc12f0fe616c54de4108f94ca566f.jpg  
      inflating: ./clothes_dataset/red_shoes/862517345bcf35aaa2f19be922f6ba0ffbecbad3.jpg  
      inflating: ./clothes_dataset/red_shoes/86f7bd3d8a367abed12e1331eae1800a69abcf85.jpg  
      inflating: ./clothes_dataset/red_shoes/86ff313985662f49467e7a89a1e42746c07d6091.jpg  
      inflating: ./clothes_dataset/red_shoes/87b8e0dbf9659cd62b1b8879e7e234863fda8b62.jpg  
      inflating: ./clothes_dataset/red_shoes/88e3fac048c00a4eddf8ade7a2b515df1b5279c0.jpg  
      inflating: ./clothes_dataset/red_shoes/896f98d1b7024bde45d59f1318288a2c2b0f0f84.jpg  
      inflating: ./clothes_dataset/red_shoes/89d7b1da4905804558fc382313d512dc0e27ed3f.jpg  
      inflating: ./clothes_dataset/red_shoes/8a2a58e6967df1210add39dcb54632f9dc69281d.jpg  
      inflating: ./clothes_dataset/red_shoes/8a73cf8994c06cf8090d75e76386d04166f6e72a.jpg  
      inflating: ./clothes_dataset/red_shoes/8a90276875d4759b2e65c2ced16e9851d327ef40.jpg  
      inflating: ./clothes_dataset/red_shoes/8b4db084176f94046ca740f3301a49d1e7fde800.jpg  
      inflating: ./clothes_dataset/red_shoes/8b73eea8c5842e4ebd8ae0d5b37feadfc013b860.jpg  
      inflating: ./clothes_dataset/red_shoes/8bea0e85f7b09ca6d57886aa67746dcf8f9deae9.jpg  
      inflating: ./clothes_dataset/red_shoes/8c2652c8af35b5bbdbeaffb44cb44762351205fd.jpg  
      inflating: ./clothes_dataset/red_shoes/8c2f00cf5eee970cbfefb6161ae5f80297923925.jpg  
      inflating: ./clothes_dataset/red_shoes/8c5429dc0597dded99767f643b878ca321a05e2e.jpg  
      inflating: ./clothes_dataset/red_shoes/8c6b0e69bc01350ea98420ca8f24a79ee41588b2.jpg  
      inflating: ./clothes_dataset/red_shoes/8cb236cc3626de89fc421f62a14c59d0ce933eb4.jpg  
      inflating: ./clothes_dataset/red_shoes/8e8e20acc7b2b0162a9c60e862a8ad71e397e210.jpg  
      inflating: ./clothes_dataset/red_shoes/8ee3841686314be479aef5f7d10546af338a95dc.jpg  
      inflating: ./clothes_dataset/red_shoes/8efddbc7fc34bcaac0ea0f47a7d19d673b033b9a.jpg  
      inflating: ./clothes_dataset/red_shoes/8f04646c34d4a9483b0c2504260f71d062ebf170.jpg  
      inflating: ./clothes_dataset/red_shoes/8f5ee8a49cd0761f5860f2dffcd21a3f594a55e9.jpg  
      inflating: ./clothes_dataset/red_shoes/8fb3866358f1c1adb435a1c9ae4530b7255a0371.jpg  
      inflating: ./clothes_dataset/red_shoes/90146df8c78f26998529bb53f9837ecfeca515e4.jpg  
      inflating: ./clothes_dataset/red_shoes/9015e586d629a4e575c067ed2a9af604e27d948c.jpg  
      inflating: ./clothes_dataset/red_shoes/90807fd0467350b2a08af6fe4272f7f391535f33.jpg  
      inflating: ./clothes_dataset/red_shoes/9126a7bb32703ca17114b9c8f6a546469fdc4d50.jpg  
      inflating: ./clothes_dataset/red_shoes/9148d164206479c4a52345b83d561c420bf57e44.jpg  
      inflating: ./clothes_dataset/red_shoes/919e8b0a6859fd24d52c062fc6e2f7f56d0ad87b.jpg  
      inflating: ./clothes_dataset/red_shoes/93b04b09bde5efc20a65c5c07a2ff197f794189e.jpg  
      inflating: ./clothes_dataset/red_shoes/93bbf69736901225dde4424b79915d17fefa95e5.jpg  
      inflating: ./clothes_dataset/red_shoes/93f25304c0d58e270d8e600eeb72f5484f438e2a.jpg  
      inflating: ./clothes_dataset/red_shoes/94647cdc1054bcde59685c927c01d2d8bdc321d3.jpg  
      inflating: ./clothes_dataset/red_shoes/9469c05d6626c928454b694cb332ab576761540b.jpg  
      inflating: ./clothes_dataset/red_shoes/94847c7b16bccb4ce3530701f5962bd36ebc059f.jpg  
      inflating: ./clothes_dataset/red_shoes/9491b2f80f24cee75865aa5181a8582e51dfa3ef.jpg  
      inflating: ./clothes_dataset/red_shoes/950cbeed0f4068dbf1967057ec29d41ccc731c90.jpg  
      inflating: ./clothes_dataset/red_shoes/957a54bcd1912b54e55188a45af2afb751347f8f.jpg  
      inflating: ./clothes_dataset/red_shoes/95818ed55df8a6dd209ecb6b43d9de4a5b886d40.jpg  
      inflating: ./clothes_dataset/red_shoes/9607a5fc59c803c2ccf1adfce0a8b04e15a16f7e.jpg  
      inflating: ./clothes_dataset/red_shoes/969ecc6cd296d2ad9e346bf7937cf9be92436af9.jpg  
      inflating: ./clothes_dataset/red_shoes/96f5d9ae18b4338e5431aad20a7030c589975658.jpg  
      inflating: ./clothes_dataset/red_shoes/970af3c5db86421633ff5161753840a61cd18b75.jpg  
      inflating: ./clothes_dataset/red_shoes/972a9b1c4c7390964471cddd979a40fb24dada33.jpg  
      inflating: ./clothes_dataset/red_shoes/984228a122ac434ff3cc7a80836517f620208221.jpg  
      inflating: ./clothes_dataset/red_shoes/985ea49de869ea1f3f12484d57cc1cdc720ecaae.jpg  
      inflating: ./clothes_dataset/red_shoes/98edfe1b3da8e33829833acdaf24069d62a7b883.jpg  
      inflating: ./clothes_dataset/red_shoes/99260ab33294886fee18305634e14251e278a9ec.jpg  
      inflating: ./clothes_dataset/red_shoes/9a1131143e82fae6332e2fd3be0a630e3b557232.jpg  
      inflating: ./clothes_dataset/red_shoes/9a3bfaee5270eb1e98678b811f6447e98b2daf13.jpg  
      inflating: ./clothes_dataset/red_shoes/9b5ad8d3b62a9cd5c15716f90b853d4414e55196.jpg  
      inflating: ./clothes_dataset/red_shoes/9bd8f95345d58961f397f5d7977c1b3c4327c9a7.jpg  
      inflating: ./clothes_dataset/red_shoes/9bfb9fc6d5f7ba79fa302dcc1f6ab5ae8da8cec6.jpg  
      inflating: ./clothes_dataset/red_shoes/9c27a8032461a7e2835d28e2bec3eac3fa13f8d4.jpg  
      inflating: ./clothes_dataset/red_shoes/9c496ddf553d669da25cee8143e3ec5522f12308.jpg  
      inflating: ./clothes_dataset/red_shoes/9ccce28e5677410a4ea2052436ec6d169ebfcbd9.jpg  
      inflating: ./clothes_dataset/red_shoes/9d3e3f7c9f687c2c8289ccde0ba42bb253bc6932.jpg  
      inflating: ./clothes_dataset/red_shoes/9db2f90d07c6d46098b4abaab86fe1a723ca17cc.jpg  
      inflating: ./clothes_dataset/red_shoes/9dc5a8be5896d61ec203427a16f92700611aa6af.jpg  
      inflating: ./clothes_dataset/red_shoes/9dc6758e0b2357cb85a94b3121d96105e0b23577.jpg  
      inflating: ./clothes_dataset/red_shoes/9e44c44e9fa714912d78d0c5ce5876dcc1f2771d.jpg  
      inflating: ./clothes_dataset/red_shoes/9efb3964712e65e0a2b1c1d7a558f34785dff707.jpg  
      inflating: ./clothes_dataset/red_shoes/9f5ac8686cc37ec40f4c4852dca6579bd84ac4ab.jpg  
      inflating: ./clothes_dataset/red_shoes/9f828f98d8bed3784379faa2dac2b2ab9b3c2045.jpg  
      inflating: ./clothes_dataset/red_shoes/a021a1a196d81df2e34e532047ccf9b5d308d36c.jpg  
      inflating: ./clothes_dataset/red_shoes/a0e648044c98bb6d4435cd4bf8260d29efc3421a.jpg  
      inflating: ./clothes_dataset/red_shoes/a0f7c8811902133ba65e3cbc1c6135f4cce6f62b.jpg  
      inflating: ./clothes_dataset/red_shoes/a1628d1d47b178b3a53c8e8ce14bb8132bdd1437.jpg  
      inflating: ./clothes_dataset/red_shoes/a32060e11ecf74d23393102151325e7c826b4cfb.jpg  
      inflating: ./clothes_dataset/red_shoes/a3310f8196e816ec3946e07f8417a789f0649ae8.jpg  
      inflating: ./clothes_dataset/red_shoes/a43d1ed464dbba7b90acae9c8a92687b8d27c351.jpg  
      inflating: ./clothes_dataset/red_shoes/a4462bfc0d5e4dbae499ff88ceb0d6add6a528a2.jpg  
      inflating: ./clothes_dataset/red_shoes/a44952bf9336c2fc61eb5b6868a8018777fdac02.jpg  
      inflating: ./clothes_dataset/red_shoes/a46d393a4abeb97f0b66f1dc1465640f12f5cf8d.jpg  
      inflating: ./clothes_dataset/red_shoes/a4bdf8149119ffa016113b59041855730184464e.jpg  
      inflating: ./clothes_dataset/red_shoes/a522df9bcba9161d0bb9a873e7cbf0f618542b05.jpg  
      inflating: ./clothes_dataset/red_shoes/a5673b8db384db5c2d1183cc22a4266b82491ac9.jpg  
      inflating: ./clothes_dataset/red_shoes/a5c781d1757602ea2f95027bc02107d913e670f7.jpg  
      inflating: ./clothes_dataset/red_shoes/a630f80d57727c46b21b2ef9376d0a36e389bb84.jpg  
      inflating: ./clothes_dataset/red_shoes/a6bbeec2117592a2e5f2c6fd64e7abfcbcef6f39.jpg  
      inflating: ./clothes_dataset/red_shoes/a723b016fcd2f1bfb958b3f75a02f643150adfaa.jpg  
      inflating: ./clothes_dataset/red_shoes/a746c9c64e5b669a6ca7759236cf7626ddac6546.jpg  
      inflating: ./clothes_dataset/red_shoes/a79464ebf127d229b64c648f5fb74d6d43d1172c.jpg  
      inflating: ./clothes_dataset/red_shoes/a81fb15cf32a5cc21217e5b239ef4d47f5030a52.jpg  
      inflating: ./clothes_dataset/red_shoes/a86a6742f194e02a7c9c5caef4bb7c743820327d.jpg  
      inflating: ./clothes_dataset/red_shoes/a87df99fefe43ec8867458715de3a9d306b97a67.jpg  
      inflating: ./clothes_dataset/red_shoes/a8aabd0651a62e08412cf9dec434bda4e93a668f.jpg  
      inflating: ./clothes_dataset/red_shoes/a920e15f485fefe292efa417768404cef743d868.jpg  
      inflating: ./clothes_dataset/red_shoes/a96a0510dac5d9fe2d1b7edb38e06be62b0cd6ce.jpg  
      inflating: ./clothes_dataset/red_shoes/a972535469162320a72ec2c6743c332380c92c10.jpg  
      inflating: ./clothes_dataset/red_shoes/aa6e9ea771a2f90a372f3b522451d0187c6af755.jpg  
      inflating: ./clothes_dataset/red_shoes/aace439b845ba75395b94f7231bcccc32d0f9f63.jpg  
      inflating: ./clothes_dataset/red_shoes/aaed6455c22b5bd59a27f7cfbc844d2d02ef5a0a.jpg  
      inflating: ./clothes_dataset/red_shoes/ab45fe987ebb28668461b20cf586fb4f7318f5eb.jpg  
      inflating: ./clothes_dataset/red_shoes/ab591b20d6459506cbb227008a44d5fbc938ced2.jpg  
      inflating: ./clothes_dataset/red_shoes/ab979c6fcb30118504d3e91877d0dda278279258.jpg  
      inflating: ./clothes_dataset/red_shoes/ac35e5827c25f7de6ae1475a7214c75b804e06b5.jpg  
      inflating: ./clothes_dataset/red_shoes/aceaad06bceb1f67537562287a5acd252bfc3e84.jpg  
      inflating: ./clothes_dataset/red_shoes/ad02af9ba3b66a4186eb918589b24aa9a6126c5f.jpg  
      inflating: ./clothes_dataset/red_shoes/ad5a81154fa33a106ed7492476cca470fb6519ee.jpg  
      inflating: ./clothes_dataset/red_shoes/ad9c18a6e5081bae7e5ce8afe11c127762f5048e.jpg  
      inflating: ./clothes_dataset/red_shoes/adffa60606f1fd808aff8c399d20be712aceaff7.jpg  
      inflating: ./clothes_dataset/red_shoes/af08d847b8f22862c8ad1a885969cb3ee08ec58e.jpg  
      inflating: ./clothes_dataset/red_shoes/af1934bf94e4a37383b2588621bf15e2122c56a6.jpg  
      inflating: ./clothes_dataset/red_shoes/af90081ad5004589344d9f8e8870c9e06350e0b3.jpg  
      inflating: ./clothes_dataset/red_shoes/af960e068be94c753ac5ca713bfa4decac824850.jpg  
      inflating: ./clothes_dataset/red_shoes/afad817bfdb735b685b5dd61b3cf1ca3cc3fc9d0.jpg  
      inflating: ./clothes_dataset/red_shoes/b0810035a3ab7c5b1a6470e3fd78a8dbe9e87911.jpg  
      inflating: ./clothes_dataset/red_shoes/b0dd8baff5f7b047e62c451ee53629cecf19b519.jpg  
      inflating: ./clothes_dataset/red_shoes/b13d2dac20e87e6d0e03004e0a0c2ea6df712178.jpg  
      inflating: ./clothes_dataset/red_shoes/b19b4f9a477cf849271aea4c6cbc30f3ea30fccb.jpg  
      inflating: ./clothes_dataset/red_shoes/b2bc83fedc75e493a67c802e3a25281ba952e60e.jpg  
      inflating: ./clothes_dataset/red_shoes/b40f7a0d2c1d595f74197c42cb53b5b33bc38d8f.jpg  
      inflating: ./clothes_dataset/red_shoes/b4313817a87be663327382ea1153bd7ad41d2363.jpg  
      inflating: ./clothes_dataset/red_shoes/b491513250a40ae29a76d7aff6cd5b0d11da1461.jpg  
      inflating: ./clothes_dataset/red_shoes/b492c187da028511e04a99fd17c094cb93d9e7fa.jpg  
      inflating: ./clothes_dataset/red_shoes/b4b43246a75be3fa23f7a9048b9b4cbb47f55f60.jpg  
      inflating: ./clothes_dataset/red_shoes/b4cc1c8b02df104edbb8d973a2e3a2311ce38584.jpg  
      inflating: ./clothes_dataset/red_shoes/b53d60607fae5f3b718a82dbee04c9af13622ae2.jpg  
      inflating: ./clothes_dataset/red_shoes/b59f4afd24d3dcae080d7f2613fb2c2829fabbc0.jpg  
      inflating: ./clothes_dataset/red_shoes/b61d7e0743392beca9d3c11fd047fb963082b821.jpg  
      inflating: ./clothes_dataset/red_shoes/b62092868c37e75f347fe83d325cda28b436977c.jpg  
      inflating: ./clothes_dataset/red_shoes/b668f71e939e77accbad7e0e54c5a46c6aa4d402.jpg  
      inflating: ./clothes_dataset/red_shoes/b67961fe5e019e7a2ea150f785e1bb217ec0e615.jpg  
      inflating: ./clothes_dataset/red_shoes/b795326795050b0a2dbdf298dcc4a4e6b67eaecb.jpg  
      inflating: ./clothes_dataset/red_shoes/b7d5e64cf6bc2d54d7a9096cf96c3b50c74a0988.jpg  
      inflating: ./clothes_dataset/red_shoes/b7d8b3ac5ddc47cbc0c7a745398294b165e4c5e0.jpg  
      inflating: ./clothes_dataset/red_shoes/b873201b9d5ffd0d9ca288c67ccaeff3f157ebfa.jpg  
      inflating: ./clothes_dataset/red_shoes/b889bd76949696ebef851772cdc44210ced1cf88.jpg  
      inflating: ./clothes_dataset/red_shoes/b8e728fa5166b23ee96e2fbce1fe8dcc82ca84dc.jpg  
      inflating: ./clothes_dataset/red_shoes/b924ee67c86511002595ceca7a46544e79f45247.jpg  
      inflating: ./clothes_dataset/red_shoes/b9449a320df24174f619398fb7896e2e439a0245.jpg  
      inflating: ./clothes_dataset/red_shoes/b9d2fc66b42652d4e69302d7a8b72307303f40a7.jpg  
      inflating: ./clothes_dataset/red_shoes/b9e264017f9294428f04e016217e9016f87bc93a.jpg  
      inflating: ./clothes_dataset/red_shoes/b9eafadb602a235c0c91aca9ccd2855191cb8f38.jpg  
      inflating: ./clothes_dataset/red_shoes/b9f5bbf65a61a9d62eb445112fcfba8512114f74.jpg  
      inflating: ./clothes_dataset/red_shoes/ba8cc90eb09abbbea48dc85f7ef59def1aea540f.jpg  
      inflating: ./clothes_dataset/red_shoes/ba9e00a797ee4e05af643bbd6856ad1d7ef681fb.jpg  
      inflating: ./clothes_dataset/red_shoes/bab1c03db4694635f73a75f188d4cce39e2b8126.jpg  
      inflating: ./clothes_dataset/red_shoes/bae6d046e64031646d42ce080b4c7a3dcaa3ef89.jpg  
      inflating: ./clothes_dataset/red_shoes/bb47ad3a1cfe4e07b000b7a187fbdee4548e059a.jpg  
      inflating: ./clothes_dataset/red_shoes/bb4b1b7883aa62056ebe1445d5b1b431e07926c2.jpg  
      inflating: ./clothes_dataset/red_shoes/bbb27bc9bf81d81bbf4e12a7d86cb18a523567a3.jpg  
      inflating: ./clothes_dataset/red_shoes/bc482309b0ba0b4a0b8ac1af710257fc3df7b797.jpg  
      inflating: ./clothes_dataset/red_shoes/bc8fce85f808b9750657c5792ed4898ef7bf0fdb.jpg  
      inflating: ./clothes_dataset/red_shoes/bd56ad45f3b6f335c531eb5842e629515a71b2ee.jpg  
      inflating: ./clothes_dataset/red_shoes/be602b5e1c8e07d5133747e72e096f340fb0c952.jpg  
      inflating: ./clothes_dataset/red_shoes/be8a493c87b383b3541d72b6ea1b3711fd2525bf.jpg  
      inflating: ./clothes_dataset/red_shoes/be920684b5f896ae8fb6b310c4311345e39a4481.jpg  
      inflating: ./clothes_dataset/red_shoes/bed060cd2d1f495a649b54c783c0b743f7a4cd3a.jpg  
      inflating: ./clothes_dataset/red_shoes/bfb838b7da4e4c3d09e507256b21660c16b2cd9e.jpg  
      inflating: ./clothes_dataset/red_shoes/c060ea216f258a3104ce894b27a144b9c4ecf05e.jpg  
      inflating: ./clothes_dataset/red_shoes/c0b12bac683fcb87fbed23beec79456d468cf806.jpg  
      inflating: ./clothes_dataset/red_shoes/c0dd384ceae4c39e723e751b738d955fc34610f4.jpg  
      inflating: ./clothes_dataset/red_shoes/c0de8128e9894f8f66e1fd3dc4d3402455d9f3ef.jpg  
      inflating: ./clothes_dataset/red_shoes/c180f64d795fa2c75963f1cb7b231273fb8c7799.jpg  
      inflating: ./clothes_dataset/red_shoes/c183a48e50f674983427d6dd080670e277890f20.jpg  
      inflating: ./clothes_dataset/red_shoes/c19d42e1c6ac404487a26af1882bd3916b02e75f.jpg  
      inflating: ./clothes_dataset/red_shoes/c1ade7d60612028dae6c9844c97adb0a21fac574.jpg  
      inflating: ./clothes_dataset/red_shoes/c23f9fcb3caebad169fd4b671cf71fd196fed7e3.jpg  
      inflating: ./clothes_dataset/red_shoes/c2695bab320c3a44c468bc7dc49c0d1e06647126.jpg  
      inflating: ./clothes_dataset/red_shoes/c272f20e3745de9cc345dbd0af44d8ac3291fac2.jpg  
      inflating: ./clothes_dataset/red_shoes/c2d6d8b6147a17d106b019a383dd96371b0d0f89.jpg  
      inflating: ./clothes_dataset/red_shoes/c35978ad37259988e4d26cb9eea49d6ba17f3f4e.jpg  
      inflating: ./clothes_dataset/red_shoes/c38427eb7e28b18b61f6643fe4bd514ba9f7745c.jpg  
      inflating: ./clothes_dataset/red_shoes/c3a3e3d528c78d1f2cb5f90c58891683a621f304.jpg  
      inflating: ./clothes_dataset/red_shoes/c422fe5bf1000b01785ed6aefb62a7844ff28d73.jpg  
      inflating: ./clothes_dataset/red_shoes/c43113bc63a11590bb27f117e80582594838e5b0.jpg  
      inflating: ./clothes_dataset/red_shoes/c57471e1fbc4492360e3167a4e38e6af246a12f9.jpg  
      inflating: ./clothes_dataset/red_shoes/c5e2037829c1c0545b0661ff74f599e11392ff51.jpg  
      inflating: ./clothes_dataset/red_shoes/c63205ae651d880aa090031e3e00248b58cff469.jpg  
      inflating: ./clothes_dataset/red_shoes/c71c8483cc5253815ae13717ff856dbe64f1eab3.jpg  
      inflating: ./clothes_dataset/red_shoes/c78e07a2382ee599b01c9c4036ebff6515d3f396.jpg  
      inflating: ./clothes_dataset/red_shoes/c7bbd21921d37aab43d2724565fdb72710890b1e.jpg  
      inflating: ./clothes_dataset/red_shoes/c7c2c16b2d06a6c1e59254ac7937412984af8ff4.jpg  
      inflating: ./clothes_dataset/red_shoes/c80b98852ba7a5212dd8d0268f002b0eb4522b8c.jpg  
      inflating: ./clothes_dataset/red_shoes/c88495176978d60a76dd6d79137776cd21c8d272.jpg  
      inflating: ./clothes_dataset/red_shoes/c9b0224db9e30f6e98301a507c0ddc2d09825e66.jpg  
      inflating: ./clothes_dataset/red_shoes/ca2e59752dd5e8d859cc3d1064fcf58d44373752.jpg  
      inflating: ./clothes_dataset/red_shoes/cb65d6c0fdf52a38a57675999cd0e7cd23d17abf.jpg  
      inflating: ./clothes_dataset/red_shoes/cc18ac4985beb0e2d2b16653657f934ec75263c6.jpg  
      inflating: ./clothes_dataset/red_shoes/cc431127e9ee4e5ee3af6f37ff3390bce7009cab.jpg  
      inflating: ./clothes_dataset/red_shoes/cc6919db1d0853da7b40a76035333f583b743e08.jpg  
      inflating: ./clothes_dataset/red_shoes/cced210e572d272b6cf6a9d4db4ac5ed97279684.jpg  
      inflating: ./clothes_dataset/red_shoes/cd12c7b1b92fb9dfb001a136400aabd578524d0a.jpg  
      inflating: ./clothes_dataset/red_shoes/cd366e0bb80e5e173a3cf373324de28c0968d1a9.jpg  
      inflating: ./clothes_dataset/red_shoes/d00689d6968ac1620ff0d09829c910db351cdb20.jpg  
      inflating: ./clothes_dataset/red_shoes/d0b1d78f588699c4bc3c2c4b66967bdf6bfce6ef.jpg  
      inflating: ./clothes_dataset/red_shoes/d0e604f8ac129743bd5f797ccd221e80a26fa7e8.jpg  
      inflating: ./clothes_dataset/red_shoes/d22309290b1505e093fb776a25e616de11238340.jpg  
      inflating: ./clothes_dataset/red_shoes/d2704b3889632ad2e6564181be863745777db27b.jpg  
      inflating: ./clothes_dataset/red_shoes/d34a9c2416ea31c7e92d2bbf6bea78c973d3129b.jpg  
      inflating: ./clothes_dataset/red_shoes/d37d67acc5804589c12b8e3ce6f9fbc6908d3185.jpg  
      inflating: ./clothes_dataset/red_shoes/d3f77ec66217af5d9dc8c0e10fced56737cb3cd0.jpg  
      inflating: ./clothes_dataset/red_shoes/d4a6182cfa995986dc0214d2811d2fedecadfa6e.jpg  
      inflating: ./clothes_dataset/red_shoes/d4e4e1d25572824beb24fbed9fccf4b14030babd.jpg  
      inflating: ./clothes_dataset/red_shoes/d55cf543c415eaa5760a2e95aa5553e78f722a5a.jpg  
      inflating: ./clothes_dataset/red_shoes/d575c38fcc53596ca68ff2eebc8f8f7ac68c8a04.jpg  
      inflating: ./clothes_dataset/red_shoes/d5b9683259a17a45831d047ed0337fa76e9dd02d.jpg  
      inflating: ./clothes_dataset/red_shoes/d774f92567c5622fbad335e655522e0d9010e99c.jpg  
      inflating: ./clothes_dataset/red_shoes/d7e6351acb63e5b44d0c9debaa721b9858bdbf43.jpg  
      inflating: ./clothes_dataset/red_shoes/d88e12ce52feb928c4b9bd6e47d5971fb5fcc969.jpg  
      inflating: ./clothes_dataset/red_shoes/d91b175a13496e70453b2227640264431c25c8e6.jpg  
      inflating: ./clothes_dataset/red_shoes/d93acba6412c7b0ebe5d733c0ae7b510d60dfe76.jpg  
      inflating: ./clothes_dataset/red_shoes/d9b969a9653ab535b553fcb8d014a10faa0b9f00.jpg  
      inflating: ./clothes_dataset/red_shoes/d9c262e4b623becc248d14881a142505d04c6759.jpg  
      inflating: ./clothes_dataset/red_shoes/d9c286dc1701a3d007bf84c95544d534640dbe99.jpg  
      inflating: ./clothes_dataset/red_shoes/db3a4c964ae4c81c2542fe82100904fddde9db4c.jpg  
      inflating: ./clothes_dataset/red_shoes/db5e76ab293f81c0512c5e5f895516332dcd4a88.jpg  
      inflating: ./clothes_dataset/red_shoes/db8a7a50e9993f5b465a500ac285823feb095f9b.jpg  
      inflating: ./clothes_dataset/red_shoes/dcae833a4878f3ad1bee86312cb52ed22f832c35.jpg  
      inflating: ./clothes_dataset/red_shoes/dd221a88464d32bf2b64b4a6bebe38b30ea91475.jpg  
      inflating: ./clothes_dataset/red_shoes/de207cb59878824fd9e5b979e46b561fe404a75a.jpg  
      inflating: ./clothes_dataset/red_shoes/de7ccdaf10c0b4fd2393b89940d3fe29c9a83a80.jpg  
      inflating: ./clothes_dataset/red_shoes/decba5e855eea2dfbf7973b9f3a5649e8a8f1464.jpg  
      inflating: ./clothes_dataset/red_shoes/df91b5d447d99da2d18e5f44685d30fdbb274ceb.jpg  
      inflating: ./clothes_dataset/red_shoes/dfb9a35547ae1fd556b25c434628f2ab22896b1a.jpg  
      inflating: ./clothes_dataset/red_shoes/dfcdfe45bfd8705bbe1036689816f6767f33b840.jpg  
      inflating: ./clothes_dataset/red_shoes/e01b49006fd59c21d74ef7f44ace6f1d231efbfb.jpg  
      inflating: ./clothes_dataset/red_shoes/e076621a84466edb2b912805de9876b9a3719c89.jpg  
      inflating: ./clothes_dataset/red_shoes/e0c64562f0a0aa00c35be126b39bb8828b0900d8.jpg  
      inflating: ./clothes_dataset/red_shoes/e10b512f9d75df28fdd63375dda4ce919d60550f.jpg  
      inflating: ./clothes_dataset/red_shoes/e1101dc463446623bc75cb0a627339a66eaa0701.jpg  
      inflating: ./clothes_dataset/red_shoes/e12ba3b8c12d11a2f46b9ff406f7210253c74c74.jpg  
      inflating: ./clothes_dataset/red_shoes/e1396253a12b050f267f239b39720ae6d7c2b59a.jpg  
      inflating: ./clothes_dataset/red_shoes/e1528bd5e0dccd62dd39b737e425abbdb3542988.jpg  
      inflating: ./clothes_dataset/red_shoes/e16b080208257e3f5cc11a75829848277af7138d.jpg  
      inflating: ./clothes_dataset/red_shoes/e29b3db8e778bdc869bd8ea8ba5db12a00e350e1.jpg  
      inflating: ./clothes_dataset/red_shoes/e2def826343213405e7e04934ce47b86490f6768.jpg  
      inflating: ./clothes_dataset/red_shoes/e305670a1e1ec098af8ae408c7cdbb68c3fb4177.jpg  
      inflating: ./clothes_dataset/red_shoes/e32514edcd70c83ff6c15083d1501a917b683ac2.jpg  
      inflating: ./clothes_dataset/red_shoes/e41c942fde4ba4557e1a6765b278cd040e36fe41.jpg  
      inflating: ./clothes_dataset/red_shoes/e517e4dad7c4bdf9ad54da5466f81aaf0ddc0030.jpg  
      inflating: ./clothes_dataset/red_shoes/e5eb32160a724126f579852ffa9fc2b803086574.jpg  
      inflating: ./clothes_dataset/red_shoes/e6131c7a937beb4e8e3ae117e3144b6523fe0921.jpg  
      inflating: ./clothes_dataset/red_shoes/e6b5d9247e162d211d4bc66a988b32edbd69108b.jpg  
      inflating: ./clothes_dataset/red_shoes/e6d82558a630d086160b1bfe7d30d1070a1ebda2.jpg  
      inflating: ./clothes_dataset/red_shoes/e753ded14fa05d496e40b57f122c2f2ebed834cf.jpg  
      inflating: ./clothes_dataset/red_shoes/e7d55f2e2e94b0ae54fbbb86788e40fc2fdea6b3.jpg  
      inflating: ./clothes_dataset/red_shoes/e815c5c534c63be40fd388c71768983900d683bc.jpg  
      inflating: ./clothes_dataset/red_shoes/e86e72f5b31d6a0b71aeda0ece1beaf4163ef48c.jpg  
      inflating: ./clothes_dataset/red_shoes/e89c6c1e7e1fcd0fcf0154fa311f36b533e3fcf0.jpg  
      inflating: ./clothes_dataset/red_shoes/e8c5993394cbdb29fa655199cd865388d58dcf5c.jpg  
      inflating: ./clothes_dataset/red_shoes/e8e448206022be7d6e659f023aae024e3362095a.jpg  
      inflating: ./clothes_dataset/red_shoes/e943b79670dddefd849bd762153fcfe715658eb7.jpg  
      inflating: ./clothes_dataset/red_shoes/e9d1af49826f45cf1b10b47912fe9973b57fa7e7.jpg  
      inflating: ./clothes_dataset/red_shoes/ea59db5d9ba88a5bfae90343eeb4bbd346ce9e0e.jpg  
      inflating: ./clothes_dataset/red_shoes/eaba33efaf18244bab0ac98a390e6630b6de1329.jpg  
      inflating: ./clothes_dataset/red_shoes/eb720feb19ee77ae23aee7a04ea4af7146317a27.jpg  
      inflating: ./clothes_dataset/red_shoes/ebe1081ac6e20de683ab571039a236a6827e3bba.jpg  
      inflating: ./clothes_dataset/red_shoes/edca9a2dc16ce1c2c6f48712f91f310183faa2ff.jpg  
      inflating: ./clothes_dataset/red_shoes/edd66afee8993f94e2b7a0430931e215ce169e43.jpg  
      inflating: ./clothes_dataset/red_shoes/eee20f4e0260c2f23bbeb130c6f63b031ef56c06.jpg  
      inflating: ./clothes_dataset/red_shoes/eeeb9bf1d12091fd635041edf5cd690ffd44315a.jpg  
      inflating: ./clothes_dataset/red_shoes/ef202846f5211f038d55a5e02f152e14f2d42123.jpg  
      inflating: ./clothes_dataset/red_shoes/ef5b7d17269d16e0eec2ae27c1fba3e5891f2803.jpg  
      inflating: ./clothes_dataset/red_shoes/ef693b326fbf68ae4e703b7ee90010aa058d1527.jpg  
      inflating: ./clothes_dataset/red_shoes/eff133215a9d69d35a9d41cd8be890bd6756bb7f.jpg  
      inflating: ./clothes_dataset/red_shoes/f0921b218cd193902c089b21640f30f736953db9.jpg  
      inflating: ./clothes_dataset/red_shoes/f1308d78a7f8f90f28ff918d01e17882e1a37e8e.jpg  
      inflating: ./clothes_dataset/red_shoes/f1914eb8538463c7dc1dc43d8421cf11ecacc1c5.jpg  
      inflating: ./clothes_dataset/red_shoes/f20fc213bef2a9f801cdabd4429ecdf3887270a1.jpg  
      inflating: ./clothes_dataset/red_shoes/f2f1b1fa3c39f0e26cb5fedfa93ef0fe264c9636.jpg  
      inflating: ./clothes_dataset/red_shoes/f3074c838a8e1262764bf6454eb7177190368d34.jpg  
      inflating: ./clothes_dataset/red_shoes/f31e64915d822ba48250cf695cd33b03e0bff8f5.jpg  
      inflating: ./clothes_dataset/red_shoes/f3569b371134c3bd1636b64fcdca7267b0cbd002.jpg  
      inflating: ./clothes_dataset/red_shoes/f3760c1e1f0dc1565b80899105e2b2f97b89d78d.jpg  
      inflating: ./clothes_dataset/red_shoes/f470e0a132ee397f8afc68529fe57f60191048b6.jpg  
      inflating: ./clothes_dataset/red_shoes/f4a8d3ee9ff9ce552cbbefae08d2493c198b73d9.jpg  
      inflating: ./clothes_dataset/red_shoes/f4c01daec5e8b846133f9a575e19d65f66180453.jpg  
      inflating: ./clothes_dataset/red_shoes/f4de36a19e359cfdaf7d35ff5675130ba4c11c25.jpg  
      inflating: ./clothes_dataset/red_shoes/f4ffba56d79ffb6c12dd85ab1248eeee95dd9aa4.jpg  
      inflating: ./clothes_dataset/red_shoes/f50304e208ef1d6f756558de61e09f2d346d272a.jpg  
      inflating: ./clothes_dataset/red_shoes/f575825a9af532fd080c436d7533e01f470f1782.jpg  
      inflating: ./clothes_dataset/red_shoes/f5c17989dc8c5217246658589542428d0735b29c.jpg  
      inflating: ./clothes_dataset/red_shoes/f625e549dbedae9f64c43995f43faf883f3261d2.jpg  
      inflating: ./clothes_dataset/red_shoes/f6312a93701f86883cc6b2e57b4433d57537900f.jpg  
      inflating: ./clothes_dataset/red_shoes/f6737d183efb66439f3e066f30d6df8d54cdefeb.jpg  
      inflating: ./clothes_dataset/red_shoes/f6c46d5434198bd24b755609089db93916540541.jpg  
      inflating: ./clothes_dataset/red_shoes/f754aa10f5756ec5dafd32ba00ea78eb28e5e5c5.jpg  
      inflating: ./clothes_dataset/red_shoes/f7dc938e273dfb1693cbd692cd31f724dc8a4d36.jpg  
      inflating: ./clothes_dataset/red_shoes/f7f0aabad58287eb70f2470baa0b64b8f357b5db.jpg  
      inflating: ./clothes_dataset/red_shoes/f815b0a460785f25cbbc617c2bc82f7d5d9aaa92.jpg  
      inflating: ./clothes_dataset/red_shoes/f87fdc6f41bb6296c60d35f700d44c802cb73738.jpg  
      inflating: ./clothes_dataset/red_shoes/f88ac3ca435a0b3cf1b173ab7dba2e6406bbee23.jpg  
      inflating: ./clothes_dataset/red_shoes/f88ac78b7237956d976abee1cb49b3703ca73374.jpg  
      inflating: ./clothes_dataset/red_shoes/fae18fcd7a4d69848d8e0b26953263b0eeb9b9e5.jpg  
      inflating: ./clothes_dataset/red_shoes/fb496a7476ab00a9fd85fdd2e03351f71c64518c.jpg  
      inflating: ./clothes_dataset/red_shoes/fbddfa5dea91133aa9849bb061e1c4905759ce24.jpg  
      inflating: ./clothes_dataset/red_shoes/fc2370733fad7078490a515a8b70cc0e69885873.jpg  
      inflating: ./clothes_dataset/red_shoes/fc5ff30adb576c83f627d7e039f894186473afe2.jpg  
      inflating: ./clothes_dataset/red_shoes/fd0dbb8163f5b9413df2d47068469c55e39f2e40.jpg  
      inflating: ./clothes_dataset/red_shoes/fd3940929ee006d93ee87fae94c8183c9e5dbea0.jpg  
      inflating: ./clothes_dataset/red_shoes/ff17fb8602c756a708c43db8bb5ecc588225ecc9.jpg  
      inflating: ./clothes_dataset/red_shoes/ff298e7dde10b5a082b6e8f475c9f0c5444f9958.jpg  
      inflating: ./clothes_dataset/red_shoes/ff45374285c74e506c91dc6b8431b8e0bc50d146.jpg  
      inflating: ./clothes_dataset/white_dress/0081dcfbb42328514ddcae301133dc74339c737c.jpg  
      inflating: ./clothes_dataset/white_dress/0082b2e48470e949d1b0f02c47226eb63096cb78.jpg  
      inflating: ./clothes_dataset/white_dress/0218dd17779056c20216fdffb174013d41a87619.jpg  
      inflating: ./clothes_dataset/white_dress/02303eecea254b1e31189815a755d7afb453b82d.jpg  
      inflating: ./clothes_dataset/white_dress/02482143ce718697034f92937de446ca7afae887.jpg  
      inflating: ./clothes_dataset/white_dress/025ed996f97f099fd93ee3dc9e6fdcbbb95ded31.jpg  
      inflating: ./clothes_dataset/white_dress/031d92f345207371af33c6d2316d189620b739db.jpg  
      inflating: ./clothes_dataset/white_dress/033bc42853ada494d0772a59c923d0d11e987bfd.jpg  
      inflating: ./clothes_dataset/white_dress/0343ca550c3e4630b32328ec0bd633b19a6ff53c.jpg  
      inflating: ./clothes_dataset/white_dress/03a1622e01cc53d66f0ab10990658708890831ed.jpg  
      inflating: ./clothes_dataset/white_dress/03c9320a303585ea96fb997d31714fead572c5b2.jpg  
      inflating: ./clothes_dataset/white_dress/03f47ee6a09f38bc6949b7fe65bcd793315d3e87.jpg  
      inflating: ./clothes_dataset/white_dress/0437f10521585fe2ef4c4df94b8c0532c09a1285.jpg  
      inflating: ./clothes_dataset/white_dress/04d9d49c618ca7484d78d749106e08bddf6a32a0.jpg  
      inflating: ./clothes_dataset/white_dress/04ee9374cb2dc81b5dc262da740a49caa7efa330.jpg  
      inflating: ./clothes_dataset/white_dress/058a267930c4f4742ed5683c739fb15e5624f05a.jpg  
      inflating: ./clothes_dataset/white_dress/058a6cb0bd2a76786ce07b0d9a462abeb805969d.jpg  
      inflating: ./clothes_dataset/white_dress/05e59ed80eac71d106a302a94393ad509499049e.jpg  
      inflating: ./clothes_dataset/white_dress/05e90f40bd9de2058a89b8ff057be27ef0661538.jpg  
      inflating: ./clothes_dataset/white_dress/0601468fc21744cab60e258a68d0281b5d4c0cee.jpg  
      inflating: ./clothes_dataset/white_dress/0630b9a4fb88b11e450be2885969b280bfcb78d5.jpg  
      inflating: ./clothes_dataset/white_dress/064afbae09a970a768a17553b97c473806866890.jpg  
      inflating: ./clothes_dataset/white_dress/06530d3261aa994bc22e848420517b861b39544e.jpg  
      inflating: ./clothes_dataset/white_dress/0673338c158b24a543b1c09648fb4b318008f862.jpg  
      inflating: ./clothes_dataset/white_dress/06dd6e1708aa1287ac29a9207b29154d21c0007f.jpg  
      inflating: ./clothes_dataset/white_dress/073db4d536f3d9168edf8b628089b0a734da655a.jpg  
      inflating: ./clothes_dataset/white_dress/078c90696379dd1ef5023a83503f1f6badc888dc.jpg  
      inflating: ./clothes_dataset/white_dress/079dd9c37fd3be5b45d697858d3f823e5f9eab0e.jpg  
      inflating: ./clothes_dataset/white_dress/07d65a53571df3ad1ad9b66dd03a061df0bbeaea.jpg  
      inflating: ./clothes_dataset/white_dress/0877039cbf77350d288ae4443c01a2d4d179bdb1.jpg  
      inflating: ./clothes_dataset/white_dress/08f845b946009a0f5a68417f6335a9e287d099b3.jpg  
      inflating: ./clothes_dataset/white_dress/094183a4f3f0bf583831db8cec83d8ebe9d4dd13.jpg  
      inflating: ./clothes_dataset/white_dress/0949bc2451e384704692f172da442398d062029e.jpg  
      inflating: ./clothes_dataset/white_dress/09f5050530fce719e575482d68f82a18576e9399.jpg  
      inflating: ./clothes_dataset/white_dress/0a2660afa14105840297317767c13ab23d5b217b.jpg  
      inflating: ./clothes_dataset/white_dress/0a2a8c7d9d9d932febe430d0a30ca82736fc6a21.jpg  
      inflating: ./clothes_dataset/white_dress/0aa3546bea5440feaa89ff44382851f5553343c0.jpg  
      inflating: ./clothes_dataset/white_dress/0aaa34f91e22e721ed1d7e2e1553cd4626cb70a6.jpg  
      inflating: ./clothes_dataset/white_dress/0abc04cc31dd6498f5e98333dfe489f60331cda4.jpg  
      inflating: ./clothes_dataset/white_dress/0b045120f0740eef03ee73fa0f969f9dbfd6cd98.jpg  
      inflating: ./clothes_dataset/white_dress/0b2aad1d0ff2b5a050b500613e3eaeec8d2e17a9.jpg  
      inflating: ./clothes_dataset/white_dress/0b72b0f3d9579a58d2daa96d888fa44e7e92fb1e.jpg  
      inflating: ./clothes_dataset/white_dress/0b78c767f0be07410447e3dba84a94e6b0aec96f.jpg  
      inflating: ./clothes_dataset/white_dress/0cabe757378b6d45c46ca098146fe717130e64c8.jpg  
      inflating: ./clothes_dataset/white_dress/0cca743725da27763ef4a185e7f92dd001cc7a57.jpg  
      inflating: ./clothes_dataset/white_dress/0d3503475af3a174663fe6a6ca990d7e2ed54b09.jpg  
      inflating: ./clothes_dataset/white_dress/0d7da6e7bd56ea3f8e3536e7baf3ecf1e7705229.jpg  
      inflating: ./clothes_dataset/white_dress/0d841bf84a7f30748347b2785f7c275a354ca7b2.jpg  
      inflating: ./clothes_dataset/white_dress/0e0d8cfcef93a7a5d1f529895f917db79172ee5a.jpg  
      inflating: ./clothes_dataset/white_dress/0e5ddfb45324bf6a1af00ec43d53019a2816c50f.jpg  
      inflating: ./clothes_dataset/white_dress/0ec3f7000d9083cb6afb1e9551cf52cb8917fcc5.jpg  
      inflating: ./clothes_dataset/white_dress/0f0114a2b30d0dd1a118b6e786bf9dac0e833272.jpg  
      inflating: ./clothes_dataset/white_dress/0f3610659409f42581f56e374cda618626ba1223.jpg  
      inflating: ./clothes_dataset/white_dress/0f5260bfd932119ed4b9ceca70a842a10adeb2ff.jpg  
      inflating: ./clothes_dataset/white_dress/0f99453756f8dd661eedb6681ebfabdf1482d737.jpg  
      inflating: ./clothes_dataset/white_dress/0fd554ac9a11d13ba772d7d606cc9b86521c1382.jpg  
      inflating: ./clothes_dataset/white_dress/0fe7d30b3eaea93457e77148ebf14749a60a4889.jpg  
      inflating: ./clothes_dataset/white_dress/1008ed96e0b212126ae690f363951797b74c91ec.jpg  
      inflating: ./clothes_dataset/white_dress/101b95c6498187253f0acfc0487253c1738f6509.jpg  
      inflating: ./clothes_dataset/white_dress/103fee46a1e248528a0bd54d206c19e2f3431ffe.jpg  
      inflating: ./clothes_dataset/white_dress/1044b9fd49ef7c2a7f948d6069b80d87fac85a9d.jpg  
      inflating: ./clothes_dataset/white_dress/104ea6d3a5cf2eab5ba130df32d54941520a6143.jpg  
      inflating: ./clothes_dataset/white_dress/106cdb0e72d501e276fb255c8e08a854334de0b7.jpg  
      inflating: ./clothes_dataset/white_dress/10834bcee8b4f4b371befe73054d35fd50351369.jpg  
      inflating: ./clothes_dataset/white_dress/1083d4cd9c54cf9343adc5ffbaeaa83453aac7f1.jpg  
      inflating: ./clothes_dataset/white_dress/11d63903ffaea02db26bdb50b38f1b58599e0684.jpg  
      inflating: ./clothes_dataset/white_dress/120ef7f2aa57353c3fc73d130d3be33be560f96f.jpg  
      inflating: ./clothes_dataset/white_dress/125b2b7372f87a8d03f29be5a5d30327057a3413.jpg  
      inflating: ./clothes_dataset/white_dress/12e9f2c73134d27483b1afb958af99620a29bc5f.jpg  
      inflating: ./clothes_dataset/white_dress/130abd7b6a46401aa443d0d32212c056b7c8cc65.jpg  
      inflating: ./clothes_dataset/white_dress/1330aa94a2a76aaad4e96a5d2bfa912936b92f0a.jpg  
      inflating: ./clothes_dataset/white_dress/13d8cd17d43111f2f785574186113314eed97ed9.jpg  
      inflating: ./clothes_dataset/white_dress/13dbd9a1411c6686cf835080ab3b31115f8b3d8f.jpg  
      inflating: ./clothes_dataset/white_dress/1449dab91f0dccb99f6fb39e2300e3b997eb0424.jpg  
      inflating: ./clothes_dataset/white_dress/14b9812a4929f3dd24b4226ec41063536ba084f7.jpg  
      inflating: ./clothes_dataset/white_dress/14ea9b67668c3a5e1c1df108805964992e18eed4.jpg  
      inflating: ./clothes_dataset/white_dress/158d603303d31261c6d86833191dd339c26a27e1.jpg  
      inflating: ./clothes_dataset/white_dress/159c26b2406512156751577e44346ab29f75cd81.jpg  
      inflating: ./clothes_dataset/white_dress/15cc6fff4e976a1517da5e8c905a81600c8273cc.jpg  
      inflating: ./clothes_dataset/white_dress/15e13e27260953abcafdaf8ae7bd4250245b7ba0.jpg  
      inflating: ./clothes_dataset/white_dress/15e45196aa65fb43c69ccf848a6e3bea2be83395.jpg  
      inflating: ./clothes_dataset/white_dress/167b96bf213439a78df6974ee5bdc415a27c26d0.jpg  
      inflating: ./clothes_dataset/white_dress/17023c4abaecb87fb299367a5265cd62511f2b0b.jpg  
      inflating: ./clothes_dataset/white_dress/174a9aca652a54800de612c34009a10fe7c200ba.jpg  
      inflating: ./clothes_dataset/white_dress/18c543fa965d0f602ea0f6923f0a8101d5b8a52f.jpg  
      inflating: ./clothes_dataset/white_dress/18d1c2ad9d0385538366bf7f7cb6e67ff3799cda.jpg  
      inflating: ./clothes_dataset/white_dress/18db564e16c7718ebaf56181370a48d69c279ad0.jpg  
      inflating: ./clothes_dataset/white_dress/18e35fd47286682999b9ad364bdd00602d865676.jpg  
      inflating: ./clothes_dataset/white_dress/18f572c254b12a2d5b1586b92ca1196d898ef37d.jpg  
      inflating: ./clothes_dataset/white_dress/1926ef87a149e13df561e15b7513927dc2301ce6.jpg  
      inflating: ./clothes_dataset/white_dress/19c5ff6d7d89b1c812aa4983bae3fca9eabde352.jpg  
      inflating: ./clothes_dataset/white_dress/19c7fb6e1515f105907edd2d09742ea0743e00de.jpg  
      inflating: ./clothes_dataset/white_dress/19fc645f0982fee23556cfdf77cc69b89d56c5e7.jpg  
      inflating: ./clothes_dataset/white_dress/1a2024562302f5338316db56ca876e0e17cf742f.jpg  
      inflating: ./clothes_dataset/white_dress/1af0208f47c10893b59aba34a8c8a46e32f8be0f.jpg  
      inflating: ./clothes_dataset/white_dress/1b1251def21755b33203f6524de1c1774b404899.jpg  
      inflating: ./clothes_dataset/white_dress/1b5ebd2ce80fdc2a53a2b9c51ffbbd60aeae02ff.jpg  
      inflating: ./clothes_dataset/white_dress/1b6c3d78e71e14d37fe69a4b2c0a5f3e41361278.jpg  
      inflating: ./clothes_dataset/white_dress/1bac841d715711a06cf10ce618951c60c5b50ba7.jpg  
      inflating: ./clothes_dataset/white_dress/1bc465b18ff6fc61c4c109b778ff1c92d9bc68b3.jpg  
      inflating: ./clothes_dataset/white_dress/1c06f5be05dc7c458017c2bfc0ac996f1e214c7b.jpg  
      inflating: ./clothes_dataset/white_dress/1ca4afca0a214669ffd8970d131488a9f836a55f.jpg  
      inflating: ./clothes_dataset/white_dress/1cad9a44427440bad7c3546e36c6a89b8aad8758.jpg  
      inflating: ./clothes_dataset/white_dress/1d549f108cdde775c0cedf77b1f5e2d1dc14e95b.jpg  
      inflating: ./clothes_dataset/white_dress/1d96f5a4cb4ebdbda47f0f722341a9ea25d9dd69.jpg  
      inflating: ./clothes_dataset/white_dress/1e3699a87fb13ccc3f88577618175c5d40c74ae3.jpg  
      inflating: ./clothes_dataset/white_dress/1f189cf0bf4f1b1804b9d65b514c998fea411e80.jpg  
      inflating: ./clothes_dataset/white_dress/203ef3bedc98a73334795ebc7416b2a099f23113.jpg  
      inflating: ./clothes_dataset/white_dress/207d83a6f85c32f6945c9ecc4bbea01ea10e8c97.jpg  
      inflating: ./clothes_dataset/white_dress/20b396db479ac9a99b4ab43345ee9f1363e0ae30.jpg  
      inflating: ./clothes_dataset/white_dress/211603a675131dad0286feca168ed0b90f2eb8cb.jpg  
      inflating: ./clothes_dataset/white_dress/214efc40ba0561284a136dd94b06fc6fd63c50ca.jpg  
      inflating: ./clothes_dataset/white_dress/21e753a11d297c81e6748d97fd7174a2fbada242.jpg  
      inflating: ./clothes_dataset/white_dress/221a1663d34b9036298fdde977c1ccd4b8d81f8c.jpg  
      inflating: ./clothes_dataset/white_dress/222102b2182685b385553b0d8fcb634f50e5e5a3.jpg  
      inflating: ./clothes_dataset/white_dress/226aeb0d4bb9a5bc3e5432a8e90fc87e079dfa2d.jpg  
      inflating: ./clothes_dataset/white_dress/22a9a0586b6afb44b874b85133f461792e8def7c.jpg  
      inflating: ./clothes_dataset/white_dress/22b7ccca7b0783350a84b64b18afaf41b6748ad6.jpg  
      inflating: ./clothes_dataset/white_dress/22ce59d0e843f9bc8efd9a0c87d08b77ba9b8f72.jpg  
      inflating: ./clothes_dataset/white_dress/23357cb39d38efc826eb2cbd2630300a0879e3a3.jpg  
      inflating: ./clothes_dataset/white_dress/237fce898be742eb6b3bb9b32bc85ec3abff76ad.jpg  
      inflating: ./clothes_dataset/white_dress/248336ba018bb551f8b46c8edc303f40e1d41e2b.jpg  
      inflating: ./clothes_dataset/white_dress/254d1cebc1b1f2ab8bd402fba62e9a8bf6ec044d.jpg  
      inflating: ./clothes_dataset/white_dress/2582b35246fa34dda6720b525cb281fbab1f1900.jpg  
      inflating: ./clothes_dataset/white_dress/25d27f53bc3918a0b9e94c309e96b9f70f7a92f4.jpg  
      inflating: ./clothes_dataset/white_dress/260bd115a3a035dfc136a995f34d23808bad6622.jpg  
      inflating: ./clothes_dataset/white_dress/266ce666e9b88e33a23f8eadcd1232df434cf979.jpg  
      inflating: ./clothes_dataset/white_dress/2690cb2f50a09074a230fce0bb96755713a103dc.jpg  
      inflating: ./clothes_dataset/white_dress/26e12f040307ac10ec98d064e355a8b0baeafeb6.jpg  
      inflating: ./clothes_dataset/white_dress/2746d1df8c1cff5207fca680230618b281de4e09.jpg  
      inflating: ./clothes_dataset/white_dress/281c2e8bd606c79bce2fca745411d9f901a74bcf.jpg  
      inflating: ./clothes_dataset/white_dress/287c03e7e4762773750f63572a22fc8004809fbe.jpg  
      inflating: ./clothes_dataset/white_dress/288f3a21993693d61c8a3a54a3bd4413cc76c949.jpg  
      inflating: ./clothes_dataset/white_dress/2896340041309c641ef826f74c49ccc6216498c8.jpg  
      inflating: ./clothes_dataset/white_dress/29b8dbf81edcd12f14e7cf1a4685c04a47d43317.jpg  
      inflating: ./clothes_dataset/white_dress/2a09635282f485d6fa97707e42369199310bdbaa.jpg  
      inflating: ./clothes_dataset/white_dress/2a10486f38861693a3143c1b1981793149e74803.jpg  
      inflating: ./clothes_dataset/white_dress/2abc210f598e801e5d4b67180e2938f80604ddb9.jpg  
      inflating: ./clothes_dataset/white_dress/2b149843b48b0284dda317648565ebe7ba5ae3f5.jpg  
      inflating: ./clothes_dataset/white_dress/2b57525d147de2b53bc0725f37500efaadd9af82.jpg  
      inflating: ./clothes_dataset/white_dress/2b6a425a418b90638768d82e8fc8c4bbb034b628.jpg  
      inflating: ./clothes_dataset/white_dress/2b9388dd9ef25d4de56baec6ab96ade5d1257389.jpg  
      inflating: ./clothes_dataset/white_dress/2bba7671b7ccd19a2e9025e1216bb9bfcaba5ee0.jpg  
      inflating: ./clothes_dataset/white_dress/2bc2d3d204fb34128b0169c844b3775b46ea9f2e.jpg  
      inflating: ./clothes_dataset/white_dress/2bd112cf19670814d43e50eb5e2a3a3b4fc0c388.jpg  
      inflating: ./clothes_dataset/white_dress/2be95d24f62314032ea48532865ff2d883bac111.jpg  
      inflating: ./clothes_dataset/white_dress/2c15304a809fcb42ec8fc11664b91baf9a9e21d1.jpg  
      inflating: ./clothes_dataset/white_dress/2c190cbef22b27fa2f453b58c41830a344428e63.jpg  
      inflating: ./clothes_dataset/white_dress/2c8056d24d2f61778cc35ff10026771a837acd7d.jpg  
      inflating: ./clothes_dataset/white_dress/2ca7aefb9b8cfa37059ffd9a0a414ee3523e6f7b.jpg  
      inflating: ./clothes_dataset/white_dress/2cbb5d3a7caeaa2aaf8c333caaa5422533f836b6.jpg  
      inflating: ./clothes_dataset/white_dress/2d14b3baa8dff94586efc1964bbf052c203454f9.jpg  
      inflating: ./clothes_dataset/white_dress/2d19d430f6a579407fdcc3ea0c92826bba612872.jpg  
      inflating: ./clothes_dataset/white_dress/2f21bc5e441eaf7a60e75497ff5093c08d7ff378.jpg  
      inflating: ./clothes_dataset/white_dress/2f8dd467323896cd8710e3b0be461e51ae4e6bec.jpg  
      inflating: ./clothes_dataset/white_dress/30407553842f02c06e4440253bd12b2fc536d7fa.jpg  
      inflating: ./clothes_dataset/white_dress/3053409a0a83ea64e7246ebc30e97c699ffccf05.jpg  
      inflating: ./clothes_dataset/white_dress/30dbb4e479a05fcc4deb3030860054144393231b.jpg  
      inflating: ./clothes_dataset/white_dress/3136dc17ad50c8c6dcfccbca2828e231e781e954.jpg  
      inflating: ./clothes_dataset/white_dress/3170709fd4a7bf360b0afb834c52b9134fddaadb.jpg  
      inflating: ./clothes_dataset/white_dress/318f455fe61c70a36cb1a760941463b3e6827b88.jpg  
      inflating: ./clothes_dataset/white_dress/319f62d27c524120a91fe3b3ae6c5d3c111e6c49.jpg  
      inflating: ./clothes_dataset/white_dress/31a26ba6b9f761a7fdb4b01c97439689df97be5f.jpg  
      inflating: ./clothes_dataset/white_dress/326af201845b8d4cc663356daac32e383a4ed269.jpg  
      inflating: ./clothes_dataset/white_dress/33d143e9e9e97e10a55743d69649963ebb4e3d3d.jpg  
      inflating: ./clothes_dataset/white_dress/34458e7a553e951619535900f5359c95f6f7e8cf.jpg  
      inflating: ./clothes_dataset/white_dress/345075313e27f56ec7309ecbb56bcaf92e28a26f.jpg  
      inflating: ./clothes_dataset/white_dress/34d93c89981d825cc8d469ca15f21cf8c86eb2b7.jpg  
      inflating: ./clothes_dataset/white_dress/34e5b58f46fcb2d5e75163db08ded1697eeb13b8.jpg  
      inflating: ./clothes_dataset/white_dress/35056d15ed9c295115b59d603271625651f6c27c.jpg  
      inflating: ./clothes_dataset/white_dress/363f65964596c6d1a37ab9b0236e4152a636756f.jpg  
      inflating: ./clothes_dataset/white_dress/3655c0ad6bb0ff385eb652bd7f2850bfc8895e65.jpg  
      inflating: ./clothes_dataset/white_dress/36db8b5b83eb7a401e2b4f6c260e204438f2789c.jpg  
      inflating: ./clothes_dataset/white_dress/37458f21c2b9bae23c04ae15090eb9b01811abec.jpg  
      inflating: ./clothes_dataset/white_dress/374da31021a74b18ff43070440b31221e32f6d1d.jpg  
      inflating: ./clothes_dataset/white_dress/3820592c2d3e0f7f5b2db4ae9c8d9fef8fcda333.jpg  
      inflating: ./clothes_dataset/white_dress/3836798af4941caa39bf7c4a8e9e94a7ff2675fe.jpg  
      inflating: ./clothes_dataset/white_dress/38810a7660a196f80edbe185267373b27b7023e1.jpg  
      inflating: ./clothes_dataset/white_dress/388e06728eb29f1e7e7ce6f8d6fcb0338c17dd33.jpg  
      inflating: ./clothes_dataset/white_dress/393d1fc476bf8e51a2f19382a1062ea6d0f63430.jpg  
      inflating: ./clothes_dataset/white_dress/39bfab5aef6e81098e84d92e81a7b49f5f66b0df.jpg  
      inflating: ./clothes_dataset/white_dress/3a40867b55f961bc9644915993ba5a308ac8ecda.jpg  
      inflating: ./clothes_dataset/white_dress/3a42a0da5fdb17adc0bd8042288656892d8529ee.jpg  
      inflating: ./clothes_dataset/white_dress/3a64f80ab1dd7d5fb0d535c68bf2b3be686c3c13.jpg  
      inflating: ./clothes_dataset/white_dress/3a6d5320aaf3eb78418dccdf25bcbfd348a25a4c.jpg  
      inflating: ./clothes_dataset/white_dress/3b356c5d39925195b817107b41662f4fd09ed8fe.jpg  
      inflating: ./clothes_dataset/white_dress/3b756d97234659a0a09e5d79ffbd0c4516eb783a.jpg  
      inflating: ./clothes_dataset/white_dress/3bb122b001015e8918e1b04f588a847729045d57.jpg  
      inflating: ./clothes_dataset/white_dress/3c379d30cfa44415aad68bc2bb90bb21f2864115.jpg  
      inflating: ./clothes_dataset/white_dress/3c5ed72f6e565e9eb600e5706f409b5c234c1080.jpg  
      inflating: ./clothes_dataset/white_dress/3c786c79a3e068c636c07564e6c87503a01df52a.jpg  
      inflating: ./clothes_dataset/white_dress/3d0b7a4da169b205891d980abd47cf5d144c1703.jpg  
      inflating: ./clothes_dataset/white_dress/3d0f2c1e3a68e65267443d9bad069a8f1178b2c9.jpg  
      inflating: ./clothes_dataset/white_dress/3d21d9bda9285928548e99001271f68cec25e208.jpg  
      inflating: ./clothes_dataset/white_dress/3d2b47f834dab52753d0393c95d3ee6f7656cd21.jpg  
      inflating: ./clothes_dataset/white_dress/3d3708fde517135ef8c0f6dd6f3b225a1805a917.jpg  
      inflating: ./clothes_dataset/white_dress/3d610d98b093250c633d91cf012d60807cd85081.jpg  
      inflating: ./clothes_dataset/white_dress/3d9521eaa702940184fa3b14511cdf9798382329.jpg  
      inflating: ./clothes_dataset/white_dress/3d9d3c9446015429fa1cc40b0c63e62ba6f5556f.jpg  
      inflating: ./clothes_dataset/white_dress/3d9dfa2d4151728de69d183810c795ff1540bdb2.jpg  
      inflating: ./clothes_dataset/white_dress/3e3f80066b75b087da37c1b74db8d3e0d034fe46.jpg  
      inflating: ./clothes_dataset/white_dress/3e9b80a6a25f0a96c5488fa88c58e1bfc403ca7d.jpg  
      inflating: ./clothes_dataset/white_dress/3f6275df21a3a15c5d30fe6d1ca72fe8c00a4d64.jpg  
      inflating: ./clothes_dataset/white_dress/3f6f6801ade47f9e5efb600a8f60d23ce63ba766.jpg  
      inflating: ./clothes_dataset/white_dress/3fa4e049f5bda1e9f8b9048851d942976ac2bed9.jpg  
      inflating: ./clothes_dataset/white_dress/4007084394e0ef02f1f4fb17bc2745db164d0578.jpg  
      inflating: ./clothes_dataset/white_dress/40608c82871fc0c678b8d774f94ebc8e20599525.jpg  
      inflating: ./clothes_dataset/white_dress/407cac6b00d638f9ad28d3909acc8db567e859f3.jpg  
      inflating: ./clothes_dataset/white_dress/4119edbd85ee4c1a1cdb0c3c170f61339b7e947a.jpg  
      inflating: ./clothes_dataset/white_dress/418c9b63239646835ad91f6f4f4958772dd1adef.jpg  
      inflating: ./clothes_dataset/white_dress/4251c08107bfc21d1357d44b7dab154b4cd08502.jpg  
      inflating: ./clothes_dataset/white_dress/430fd137b5595795c0c4bfb7a580d90fa9fab659.jpg  
      inflating: ./clothes_dataset/white_dress/436254a98ecdcaead2e00b922c4bb425d6381e26.jpg  
      inflating: ./clothes_dataset/white_dress/4370784c194a8d2f70430e658619783fe663ae05.jpg  
      inflating: ./clothes_dataset/white_dress/43bcbf006e2327c8fc2ac3fef528310c8fdddfd7.jpg  
      inflating: ./clothes_dataset/white_dress/453fbe4b9fa85d18b86b3c29b2aa529cfb2ecf46.jpg  
      inflating: ./clothes_dataset/white_dress/4541ec091c379399403e2702814e2d53592d30ae.jpg  
      inflating: ./clothes_dataset/white_dress/459b307b45f1ce3b7ad710ead3f8e1cd6010712c.jpg  
      inflating: ./clothes_dataset/white_dress/46070032f9a8e893dbbda096fdf8ed3d23e0ca53.jpg  
      inflating: ./clothes_dataset/white_dress/464261f0cd3b6803118398f386e76e6de290432f.jpg  
      inflating: ./clothes_dataset/white_dress/466320e9e6aa1bfc0066d9364312617bf0f95e84.jpg  
      inflating: ./clothes_dataset/white_dress/469afe4087ed80c99f2c71296871f928770d98f5.jpg  
      inflating: ./clothes_dataset/white_dress/48074c9ca822549c757b206d7e4c47bde0d89a20.jpg  
      inflating: ./clothes_dataset/white_dress/480931a349542e3e60588edb2d9b2cc3decb8b59.jpg  
      inflating: ./clothes_dataset/white_dress/488d4ec30a083416dc3ba9047c9673c0c030352b.jpg  
      inflating: ./clothes_dataset/white_dress/491ef13a79c7d82b4bd1b652440d20b379b87aad.jpg  
      inflating: ./clothes_dataset/white_dress/494abd09fd7c40c402e5323cf19f60548bcc326e.jpg  
      inflating: ./clothes_dataset/white_dress/49df2f3f48eca5d8106ba9df7b87bf631d70a71d.jpg  
      inflating: ./clothes_dataset/white_dress/4aed1234bcd69bdebf36b03c80e96fdcc81cef74.jpg  
      inflating: ./clothes_dataset/white_dress/4ba3adeb4d3a575557fb2a7da1f551fe0c7d658a.jpg  
      inflating: ./clothes_dataset/white_dress/4bd4b603b8088b34dde56b9a7f2856e439d97886.jpg  
      inflating: ./clothes_dataset/white_dress/4c086769bfce06a8ebd875e6c09fc7d8bee5157f.jpg  
      inflating: ./clothes_dataset/white_dress/4c9a145ad7da50dd7e59c3131ba5ada776786e49.jpg  
      inflating: ./clothes_dataset/white_dress/4cc46c25b8aea35378f8a413b91c3c7631aec07d.jpg  
      inflating: ./clothes_dataset/white_dress/4d15582f3275b864b1aa46b56efcbce21c200e11.jpg  
      inflating: ./clothes_dataset/white_dress/4db3a04f7794a6a40e617102c0169cd3babf3f01.jpg  
      inflating: ./clothes_dataset/white_dress/4e2bfff841ad9e4fd206fd88fde52c265dfd7033.jpg  
      inflating: ./clothes_dataset/white_dress/4edfc341dee7c0acb5407c5854fcdcd28dcd427e.jpg  
      inflating: ./clothes_dataset/white_dress/4f3e47743883a46aeb22c5de6a00d37c17782b76.jpg  
      inflating: ./clothes_dataset/white_dress/4f67d47efde4c6b04cbfc1b81c7c6d582ad39fa7.jpg  
      inflating: ./clothes_dataset/white_dress/4f7b5f68040472dee9aafcb43cb32e6bb8c155de.jpg  
      inflating: ./clothes_dataset/white_dress/503b4b9059c3996c3c4787d1e6a9a3f82877cac9.jpg  
      inflating: ./clothes_dataset/white_dress/50522c5e4ef1403469d652e539be0d479211d361.jpg  
      inflating: ./clothes_dataset/white_dress/506acadf19d1e011f4b479e1f3bd9fd84a63967b.jpg  
      inflating: ./clothes_dataset/white_dress/50aa9ba4d56a07a27bd17caee4e6e1b1411a221b.jpg  
      inflating: ./clothes_dataset/white_dress/50be357d64269af81143b1affd3d01ff7a97ace1.jpg  
      inflating: ./clothes_dataset/white_dress/511c962d72cf32edd73b39cfc5daddee33feba70.jpg  
      inflating: ./clothes_dataset/white_dress/519922fc64d7b5f7f5b07003b54db7d7eb6a7018.jpg  
      inflating: ./clothes_dataset/white_dress/51e3a2e42107315e0a635c566f82e831c3b5da76.jpg  
      inflating: ./clothes_dataset/white_dress/52056c31ff1e76900bbcf8cfd3f6379f8213c75c.jpg  
      inflating: ./clothes_dataset/white_dress/52cb7739a732c78b9c2b414d986423c28f138e6f.jpg  
      inflating: ./clothes_dataset/white_dress/53ba0618a2a1e9933d433c1d94fdd042a72b5363.jpg  
      inflating: ./clothes_dataset/white_dress/53eb321526b35f61bf928acaf281d1e2e4abd1af.jpg  
      inflating: ./clothes_dataset/white_dress/543cb49efa9a7a1d89d634f13f550e934466cdc2.jpg  
      inflating: ./clothes_dataset/white_dress/54652a91d57cf5c109b7ccac88aef3da02704439.jpg  
      inflating: ./clothes_dataset/white_dress/551373c80717c5b0560ab25139262203bc6043f6.jpg  
      inflating: ./clothes_dataset/white_dress/5523d5f4dac5a3fb928b930295944e28923d3699.jpg  
      inflating: ./clothes_dataset/white_dress/55315353bfbc22838b88c409a1398dcd8e845963.jpg  
      inflating: ./clothes_dataset/white_dress/558a620186c6264e10b07f2a0174fd7eb6d29538.jpg  
      inflating: ./clothes_dataset/white_dress/57080cd37f206bfe8869d3a38b21c8bf502e4777.jpg  
      inflating: ./clothes_dataset/white_dress/572c3dc270a89af7824f95a8251033f78772e100.jpg  
      inflating: ./clothes_dataset/white_dress/57e50b28ce23fc9d251afcaef4f213402ec9f4d2.jpg  
      inflating: ./clothes_dataset/white_dress/57febddab2d18c9e12f9d7e92cb0a209d8691aef.jpg  
      inflating: ./clothes_dataset/white_dress/581744414275213b5779c88ab7513d1f4032a1a7.jpg  
      inflating: ./clothes_dataset/white_dress/5826c9849bfe32e220202b69fa1dff72ffe3abf7.jpg  
      inflating: ./clothes_dataset/white_dress/595a60be6c566fb70f0895661d595aa1dc2f81d0.jpg  
      inflating: ./clothes_dataset/white_dress/597afca7eb8bb2c8ea3e574a9af64d5b5d7471be.jpg  
      inflating: ./clothes_dataset/white_dress/5981031609dade9a3e477527fe59b8827bde9e2b.jpg  
      inflating: ./clothes_dataset/white_dress/59ef36b83750a17aaf1ff094cd38b43dce3c9b78.jpg  
      inflating: ./clothes_dataset/white_dress/5a064d29c31214c9e660ef5e18829534c192a915.jpg  
      inflating: ./clothes_dataset/white_dress/5a19bf26b98d73d0ddf5ddf38e463c4e85998f8d.jpg  
      inflating: ./clothes_dataset/white_dress/5a37f5603e44a4bb80dc05afa3cff10aab2cf48a.jpg  
      inflating: ./clothes_dataset/white_dress/5a5f085f67e2a32c7d71ac009fdbea3273f0ee1f.jpg  
      inflating: ./clothes_dataset/white_dress/5a6b1337f99f7c5666bff189227b267c3001361c.jpg  
      inflating: ./clothes_dataset/white_dress/5b90a50d198c33f6544523de03b24358af2da623.jpg  
      inflating: ./clothes_dataset/white_dress/5c3818ac2cc4e1c02bd447906f1d7eb73af8b230.jpg  
      inflating: ./clothes_dataset/white_dress/5c41e44186df7ad2c51c725a9706d6daf37e582f.jpg  
      inflating: ./clothes_dataset/white_dress/5d601cfd856814d354bcb11693eeb9c435e77309.jpg  
      inflating: ./clothes_dataset/white_dress/5d63037300ce0d161109d169d041e9a8f697dd25.jpg  
      inflating: ./clothes_dataset/white_dress/5dbfe59e18fe71f1961187a25eb468a877ff35b1.jpg  
      inflating: ./clothes_dataset/white_dress/5de7a0f148c61aa618022238370876767900169e.jpg  
      inflating: ./clothes_dataset/white_dress/5dfea7f4e0c29953e7bfebdcca7e6763d6de14b4.jpg  
      inflating: ./clothes_dataset/white_dress/5e4e58ad29c3966381b133d26544d54f7221f60d.jpg  
      inflating: ./clothes_dataset/white_dress/5ed29f535f13ba7e7745590313d021ece601b6d7.jpg  
      inflating: ./clothes_dataset/white_dress/5f01c1c803adac1686ba74e5595a1936c6b99a21.jpg  
      inflating: ./clothes_dataset/white_dress/5f0818b9daad8cfcd18e58c081b3fe806d52160d.jpg  
      inflating: ./clothes_dataset/white_dress/5febbb4e145d099b13180acf8c50f6030f2a311c.jpg  
      inflating: ./clothes_dataset/white_dress/604395f1155fa0b60939b5631ef74e139d65e097.jpg  
      inflating: ./clothes_dataset/white_dress/606d8c3db0eb4a55d36db0c18c49397a975c5c82.jpg  
      inflating: ./clothes_dataset/white_dress/61659aaf0661498555602973e8b145adaf8e0a13.jpg  
      inflating: ./clothes_dataset/white_dress/61837203df8523c3e8a017c73d970395f686f580.jpg  
      inflating: ./clothes_dataset/white_dress/6201d9a4f49b90ab0817a5e1471ff2229f2b25da.jpg  
      inflating: ./clothes_dataset/white_dress/6233521d99656ab4f9281751687325e8d52c7ed5.jpg  
      inflating: ./clothes_dataset/white_dress/626909a580ba35e0581cb9ab23bad4105aaf9eb1.jpg  
      inflating: ./clothes_dataset/white_dress/634e77df21cb4a3dea506785e2d6fb834c142752.jpg  
      inflating: ./clothes_dataset/white_dress/6353445804583c5f4f4e3f0500a52cdd6b03ab0d.jpg  
      inflating: ./clothes_dataset/white_dress/63e6976636190c483038a51548feada9c66038fd.jpg  
      inflating: ./clothes_dataset/white_dress/63ed2b33d265e53a408b60ec360c0a1ac196a94a.jpg  
      inflating: ./clothes_dataset/white_dress/644312f4caa73864d4701ca9973ba32b0e9fb5eb.jpg  
      inflating: ./clothes_dataset/white_dress/647288ee1fcc037eb1bf889ae11e4f981df5625b.jpg  
      inflating: ./clothes_dataset/white_dress/64819cb3cca838030e41c0075d7d375ee5a5620e.jpg  
      inflating: ./clothes_dataset/white_dress/64aa8ae1987f7c5b7d72fb484165cf10d0f443da.jpg  
      inflating: ./clothes_dataset/white_dress/64dd63c48483d86f2fb08e009e4eae11bb2928e5.jpg  
      inflating: ./clothes_dataset/white_dress/6500fbb0ad5a4c1c1d096524bafb052de04ee477.jpg  
      inflating: ./clothes_dataset/white_dress/650deec9ed62bc7def2fdd05deef58c2052897f5.jpg  
      inflating: ./clothes_dataset/white_dress/65539c03c1e6a62a17f8af671dbd732a403101dc.jpg  
      inflating: ./clothes_dataset/white_dress/65cbc28ce8112d865a3edf782e74a16a56210da2.jpg  
      inflating: ./clothes_dataset/white_dress/65db5f387a0bf7695560e33cd777fc6b35419a66.jpg  
      inflating: ./clothes_dataset/white_dress/664ba63fe6d9749b0cc078cc52bd9d6befc059b9.jpg  
      inflating: ./clothes_dataset/white_dress/66a6daebfc85e7a04ca16f539ce9a964f313c599.jpg  
      inflating: ./clothes_dataset/white_dress/671500e14a0dbd2643a7f8622d371a045b1c436e.jpg  
      inflating: ./clothes_dataset/white_dress/679d9151eb2c886a22cd58b161b183a7952aeff1.jpg  
      inflating: ./clothes_dataset/white_dress/67a4c9f333a4333519e1a7a4afffd9146c904a60.jpg  
      inflating: ./clothes_dataset/white_dress/67be35243977895bc3a3df532fcbe61dcf852714.jpg  
      inflating: ./clothes_dataset/white_dress/681c6aaf3bb95a0d68baa26bdc7e809f8bc62fa5.jpg  
      inflating: ./clothes_dataset/white_dress/687644323f17468296c0474beaf20776ed9dd3e1.jpg  
      inflating: ./clothes_dataset/white_dress/690735bc4912df3da5c2554488f5bd91d4cfedf6.jpg  
      inflating: ./clothes_dataset/white_dress/6a09b174e3297dc97379c6e8a0d6cd9688bcdcc7.jpg  
      inflating: ./clothes_dataset/white_dress/6a3b2f8edc11c03e7b63e0f37381af0158143af3.jpg  
      inflating: ./clothes_dataset/white_dress/6a9df8e7b2eeb7b5931038b2d5fcfd6d6c5388d8.jpg  
      inflating: ./clothes_dataset/white_dress/6ae76f806b0d0bf851da76dd8b24393b188a4bff.jpg  
      inflating: ./clothes_dataset/white_dress/6b79e34e15e744cbe4620f36e406d7d7d3ce9033.jpg  
      inflating: ./clothes_dataset/white_dress/6bb8ca6e5cfdaea91a9ffd402f163ed821cbba53.jpg  
      inflating: ./clothes_dataset/white_dress/6bc1b5e94bcb83d2b1feb7b33823469bb25bd740.jpg  
      inflating: ./clothes_dataset/white_dress/6bc67aeb3443764e43f49356f2c38af47b50da14.jpg  
      inflating: ./clothes_dataset/white_dress/6bd567e591c1903a8b08e05c84ab199d9213342c.jpg  
      inflating: ./clothes_dataset/white_dress/6c36a5785d701024be71740c81e88159a0d13417.jpg  
      inflating: ./clothes_dataset/white_dress/6c3ac96d5acf00ff439a343a9d5b40dbd0530a65.jpg  
      inflating: ./clothes_dataset/white_dress/6c7706d4aa5ac0ab4ebe2676758ed4e26537fa6f.jpg  
      inflating: ./clothes_dataset/white_dress/6c8963682a83d1efbaa4f572f0d05d5edc55a018.jpg  
      inflating: ./clothes_dataset/white_dress/6d58dbc5ff706776fcb4a6aa346013d0dbebe127.jpg  
      inflating: ./clothes_dataset/white_dress/6d593d1fae6b4deba59279209dd5a61d8f646279.jpg  
      inflating: ./clothes_dataset/white_dress/6d814959d92b0ac1402b0b2bf3aacf7e8951e0f6.jpg  
      inflating: ./clothes_dataset/white_dress/6df879f3888a397577c2a37ae27147d47ed5b040.jpg  
      inflating: ./clothes_dataset/white_dress/6dfbcb3fa9e9714bc67259b5eea146aa2e7c79c3.jpg  
      inflating: ./clothes_dataset/white_dress/6e19bb9d4d5261b4753671b8d31e519a5629104f.jpg  
      inflating: ./clothes_dataset/white_dress/6e2b8f4b0e3c91e92760a7e8206e1cdb9ce1daec.jpg  
      inflating: ./clothes_dataset/white_dress/6e929c3a938abcae5009933450a78e246f1ab922.jpg  
      inflating: ./clothes_dataset/white_dress/6ea274152da8b9309ddf0ae7cdbd18efcd80312c.jpg  
      inflating: ./clothes_dataset/white_dress/6eaf6f6cfce725a240f41a682afe85a6ff1af83b.jpg  
      inflating: ./clothes_dataset/white_dress/6eb54c8b83d1403129fc1a282d59b0f92032e1b9.jpg  
      inflating: ./clothes_dataset/white_dress/6f4ab24cee0b564eec9290b5991a310ec481c0c6.jpg  
      inflating: ./clothes_dataset/white_dress/705a66f50ecabe12c56016e74ca648fc3c2b6953.jpg  
      inflating: ./clothes_dataset/white_dress/70650c0464a0ef4adb799bf7adf2117bc26e8c6b.jpg  
      inflating: ./clothes_dataset/white_dress/70ad7d8ef6d6091155b69875377b24733b3fccb9.jpg  
      inflating: ./clothes_dataset/white_dress/70e3498e0c53245c72d0aa4b94dc9646253dd5d7.jpg  
      inflating: ./clothes_dataset/white_dress/7122e6eb104eb0c7009cfb49664864f9d2651ed8.jpg  
      inflating: ./clothes_dataset/white_dress/71b91a6fcddaef4d29e9fc3d04e35be350724db4.jpg  
      inflating: ./clothes_dataset/white_dress/71c6491cb5109f6680e469345ce32f77beba9392.jpg  
      inflating: ./clothes_dataset/white_dress/71dbb8d09d8215aa4db4fe260706987cdf592454.jpg  
      inflating: ./clothes_dataset/white_dress/7311a7ba7dd61edcdf90214069025dbed5cb1c55.jpg  
      inflating: ./clothes_dataset/white_dress/734c011e57841b27cc333d9df5d28bc3a2d21d80.jpg  
      inflating: ./clothes_dataset/white_dress/73518ce3784a843c5b10e8841197610817691fa2.jpg  
      inflating: ./clothes_dataset/white_dress/73953c5dd2e42c49fa764930870f53898a8965da.jpg  
      inflating: ./clothes_dataset/white_dress/73f2573a131c6961c061ecb2a2bb7bffd0108cba.jpg  
      inflating: ./clothes_dataset/white_dress/743b960ab6bba18edc0286067fc3427a5d75cd67.jpg  
      inflating: ./clothes_dataset/white_dress/749f264748f744f514c4fa4afb80a35b8d39230f.jpg  
      inflating: ./clothes_dataset/white_dress/74aed21f09e7aadbe29191852fb8c8973ebec269.jpg  
      inflating: ./clothes_dataset/white_dress/75345efcfb26d39a6a4ff0571eedf275fd6fd0eb.jpg  
      inflating: ./clothes_dataset/white_dress/76efd341412e523c52c2ea3521e936b0ff35bf55.jpg  
      inflating: ./clothes_dataset/white_dress/770fa83ca1250801267122547775b21e45a02506.jpg  
      inflating: ./clothes_dataset/white_dress/773922ce83459ddbd3f0fb07be33efa6a8b0fb58.jpg  
      inflating: ./clothes_dataset/white_dress/775b4b27a57c52afa6a640d1ff6ba2c9ca02befb.jpg  
      inflating: ./clothes_dataset/white_dress/77a5b0ebe34c95a129ba8430d3087b337e1d1b7d.jpg  
      inflating: ./clothes_dataset/white_dress/78aa8cb359c4f4f092551416148a3cfcede003ff.jpg  
      inflating: ./clothes_dataset/white_dress/78f04cd0ac5baf293881e1c3ce1de43d735456f2.jpg  
      inflating: ./clothes_dataset/white_dress/79b729f9d1d5a32043734e15a38c169abec5314a.jpg  
      inflating: ./clothes_dataset/white_dress/79e28c0c17fd9c90c6fcb9f5a1b28bfb5afcb4e0.jpg  
      inflating: ./clothes_dataset/white_dress/7a933e1a1930303158f83ddef827d2d29e3668e4.jpg  
      inflating: ./clothes_dataset/white_dress/7aa6aceb759ba0a89d62f820515939f1b2744788.jpg  
      inflating: ./clothes_dataset/white_dress/7ae3758292fff621077a29c2adcb0439f41572e7.jpg  
      inflating: ./clothes_dataset/white_dress/7b01a24e1d8190e5fda4af7e250860f969407882.jpg  
      inflating: ./clothes_dataset/white_dress/7b2c4ccf9939b99ab53d523d47982693f718c9cd.jpg  
      inflating: ./clothes_dataset/white_dress/7b94bd675905e38db6a6d2dc5835c5707b233ddb.jpg  
      inflating: ./clothes_dataset/white_dress/7bd531a31b77d520da92110891e3338e9369e953.jpg  
      inflating: ./clothes_dataset/white_dress/7c257da2cbda88fd5ff3dce1734701b56a81fb59.jpg  
      inflating: ./clothes_dataset/white_dress/7d2e2c2d12ccff4ee2f38a18087da682298d5250.jpg  
      inflating: ./clothes_dataset/white_dress/7d3fae9fd5e5afeaff8f4942e24bf26b4a6640a9.jpg  
      inflating: ./clothes_dataset/white_dress/7dc43187711ff6008478a68efa755bd5abffc022.jpg  
      inflating: ./clothes_dataset/white_dress/7dc532795bf9b2bee1cf8b5684d84783e854b4f1.jpg  
      inflating: ./clothes_dataset/white_dress/7e26d576ab3171d965b07e396a7c5b3262f5917d.jpg  
      inflating: ./clothes_dataset/white_dress/7e45db143308d55d77a03641d327ca39391c70a0.jpg  
      inflating: ./clothes_dataset/white_dress/7e4af8894b7d3e1b249d127553b060ae7474ab1d.jpg  
      inflating: ./clothes_dataset/white_dress/7e695a4a0bdbe63ca23d0b7eaaad2ecb3bec0825.jpg  
      inflating: ./clothes_dataset/white_dress/7f27f14c399c04d14ebb5540df0b9f7cd91c9a6c.jpg  
      inflating: ./clothes_dataset/white_dress/7f7a8d926a216e375f3ee1a14a5473dd1bb6b590.jpg  
      inflating: ./clothes_dataset/white_dress/7fb1e339184c59d803ebbe0b6ebbb89c633a9611.jpg  
      inflating: ./clothes_dataset/white_dress/7fcd66f7dc83bc510caf680550f02d402f2fb479.jpg  
      inflating: ./clothes_dataset/white_dress/8013c386143192869b99e0ed04d0ef876782c2ac.jpg  
      inflating: ./clothes_dataset/white_dress/801d2b28cf8aa955546679c29145ea6e7fc44a8c.jpg  
      inflating: ./clothes_dataset/white_dress/8049132020212e74e0e7d533790183bbe593394d.jpg  
      inflating: ./clothes_dataset/white_dress/8075466b07fad489df64276bb092d8713ce53b9a.jpg  
      inflating: ./clothes_dataset/white_dress/810b6dd9e7766fae0306d05a88671b020a12b87d.jpg  
      inflating: ./clothes_dataset/white_dress/812c8ddbbacd53a9f3c4ca86db3587994d2a1e20.jpg  
      inflating: ./clothes_dataset/white_dress/81e8aed616814a4abe2483022f90e14ccc218673.jpg  
      inflating: ./clothes_dataset/white_dress/82cdf8e0c56ad00593445e5e66932772ea736f80.jpg  
      inflating: ./clothes_dataset/white_dress/8365d18e0419568c388f3380e0cdf63f530c7062.jpg  
      inflating: ./clothes_dataset/white_dress/83ef0619ecee4d0b44bba69ce837073ec6a4afcd.jpg  
      inflating: ./clothes_dataset/white_dress/845309a1217ad6b406342c0543f5baa3c0f9f385.jpg  
      inflating: ./clothes_dataset/white_dress/84e5846a092d60eb912c2e8ee51c0f2a0874165c.jpg  
      inflating: ./clothes_dataset/white_dress/856a8faf00a4c2395b62db06d70e4ec2dee5ba8f.jpg  
      inflating: ./clothes_dataset/white_dress/859349fc59f72238c4c1b009cbcaf97c83346e82.jpg  
      inflating: ./clothes_dataset/white_dress/85a7864d16da5c3d542c88eb48aa74052a99a30e.jpg  
      inflating: ./clothes_dataset/white_dress/85bca5a69df199c1a22deee857017e0084e257bb.jpg  
      inflating: ./clothes_dataset/white_dress/8634fa1a3556ea9c1fc64faf185a2773d36aaea2.jpg  
      inflating: ./clothes_dataset/white_dress/86d86b21f74e4aafb6ba9e1e4bf036b8368e9bcb.jpg  
      inflating: ./clothes_dataset/white_dress/870f027e67ae285293ff691f8f00f7ecd3917c8e.jpg  
      inflating: ./clothes_dataset/white_dress/871b1ddf522b303119539f893077a3047a9f027e.jpg  
      inflating: ./clothes_dataset/white_dress/87591ab9e3aa8b51bfe30e30cb69d9cee975c3ce.jpg  
      inflating: ./clothes_dataset/white_dress/8767d682dc7a345b4d89b1072338efd3d482f55d.jpg  
      inflating: ./clothes_dataset/white_dress/880df3c88c37b6e6db970ef5111a7b4b855c9f4f.jpg  
      inflating: ./clothes_dataset/white_dress/8834ad8fe3fdad8d83b0d0a95554ed966ef45b8c.jpg  
      inflating: ./clothes_dataset/white_dress/88a1393686b953ae37ee9803298fc386f5e72565.jpg  
      inflating: ./clothes_dataset/white_dress/88fa6129aa0187875bb737780fb99dcc9d97da95.jpg  
      inflating: ./clothes_dataset/white_dress/895fd2fa87b6f11b963a84be57f72d5b34bd9f3f.jpg  
      inflating: ./clothes_dataset/white_dress/897c52a8b4bf1d5310bc54b9bddacc6a461a351d.jpg  
      inflating: ./clothes_dataset/white_dress/898789065768768ac79f7b8f941c30e9432cd2da.jpg  
      inflating: ./clothes_dataset/white_dress/89c170ad0e64110737adedcbe4c93998c3800f6d.jpg  
      inflating: ./clothes_dataset/white_dress/89e88b28a51bcbc9976899fe7f866a92d04c7e3c.jpg  
      inflating: ./clothes_dataset/white_dress/8a2d455001d7d99e30c10c2b56380f0b10223ff5.jpg  
      inflating: ./clothes_dataset/white_dress/8b4fe93a1c95e6f5f5725fb9443a043760a667a2.jpg  
      inflating: ./clothes_dataset/white_dress/8b5d0746ed242bd159b6ac841084b90e512b7e1f.jpg  
      inflating: ./clothes_dataset/white_dress/8b5e37dafb2fd02905427113992c4f0ba2440072.jpg  
      inflating: ./clothes_dataset/white_dress/8bcad1b70eee56d0e788ef9a4b29175044b8efc2.jpg  
      inflating: ./clothes_dataset/white_dress/8c7344eaf3ac8aec5814ca2f891f63b36057b192.jpg  
      inflating: ./clothes_dataset/white_dress/8c7fd1592b7e500a37dc323d8680628b2a23d2a0.jpg  
      inflating: ./clothes_dataset/white_dress/8c932541f1505ec56e7e11e2355cba19a506e638.jpg  
      inflating: ./clothes_dataset/white_dress/8c963fb0196b4aea476c0d70c8c41bcf4d270fd4.jpg  
      inflating: ./clothes_dataset/white_dress/8d0fea29c31644ed99891629dd368bf7483a2626.jpg  
      inflating: ./clothes_dataset/white_dress/8e6a3ede66638184fa2c73b2f4c07db643624999.jpg  
      inflating: ./clothes_dataset/white_dress/8ecb7ead5d15a8f78fdf8bc20441c5dcbefbadf6.jpg  
      inflating: ./clothes_dataset/white_dress/8ee585e11dcdafdaf0cb2dc61c327bb0854dea05.jpg  
      inflating: ./clothes_dataset/white_dress/8f3651ad8d04e7b060e3959c76f87f62b51d702d.jpg  
      inflating: ./clothes_dataset/white_dress/90054ffce39ba98f15ec1582b3f1bc5d875d8bf0.jpg  
      inflating: ./clothes_dataset/white_dress/900b283d3ba1151b43db2bfb622893885dfef255.jpg  
      inflating: ./clothes_dataset/white_dress/901cb9306f326b3ca5584963bbf137cfde0f4c1e.jpg  
      inflating: ./clothes_dataset/white_dress/904aef040a45497553bc6436ee2895fa207a99e1.jpg  
      inflating: ./clothes_dataset/white_dress/90819f343fd548176930266569cdc84c06aa5788.jpg  
      inflating: ./clothes_dataset/white_dress/90a74debf301f15ddbfd5c88e5e884957fa59411.jpg  
      inflating: ./clothes_dataset/white_dress/90fef6d0e0c8dbd276bbec701cbad8c5eecfee4a.jpg  
      inflating: ./clothes_dataset/white_dress/91336e297a3d45d419587b08ccf3161c3b088ef9.jpg  
      inflating: ./clothes_dataset/white_dress/916cf9e9a5429b32a1a4333a10ff69309ac70426.jpg  
      inflating: ./clothes_dataset/white_dress/91adb129bdab2f42339c6594705087f3d3679824.jpg  
      inflating: ./clothes_dataset/white_dress/91d6e65fc4fd7807f631365bd93d19e94afe6f99.jpg  
      inflating: ./clothes_dataset/white_dress/91eeda643b6712548a44ab7d985826b85e0fea04.jpg  
      inflating: ./clothes_dataset/white_dress/92b7594c47a05da3d6e84891b40a7811fa6ad35e.jpg  
      inflating: ./clothes_dataset/white_dress/931310219f1f9f7851cc7328fb7927210675e9bb.jpg  
      inflating: ./clothes_dataset/white_dress/9326ac97ec3c2d877ea196328213a0ce5031dac9.jpg  
      inflating: ./clothes_dataset/white_dress/93e5f9810f8d1d4b50c955480f0fd282fca7a1ba.jpg  
      inflating: ./clothes_dataset/white_dress/94b831dc7266662fdb446370dc1a707aeb1d4c8b.jpg  
      inflating: ./clothes_dataset/white_dress/952737b341ff804cb5b779cd5d2cad20bc5e5990.jpg  
      inflating: ./clothes_dataset/white_dress/9558e5286279ab3525f9c355476a169ecb55de15.jpg  
      inflating: ./clothes_dataset/white_dress/958e70310d70265da677d0fc547e020696186ffa.jpg  
      inflating: ./clothes_dataset/white_dress/95a85d0b604da1aafcb9d1ceb8188f3329929df5.jpg  
      inflating: ./clothes_dataset/white_dress/95e7f25cb0a1e219b568912165809227de8ec39c.jpg  
      inflating: ./clothes_dataset/white_dress/95e8c34095ef687fd83431531691feac7ef62b99.jpg  
      inflating: ./clothes_dataset/white_dress/962eaf82d2f28c49cb8e317c4c2d9c07451301ab.jpg  
      inflating: ./clothes_dataset/white_dress/964b48d4f40960b826c91a7c547244e62513c401.jpg  
      inflating: ./clothes_dataset/white_dress/978bda1cbd2659b7271fb80a3d8631a3508b8e6d.jpg  
      inflating: ./clothes_dataset/white_dress/97918509b3ce967fe034c8a7e30a306968fc0bc0.jpg  
      inflating: ./clothes_dataset/white_dress/97bed04e88218235e4da66a6cdaa88acb8487f3a.jpg  
      inflating: ./clothes_dataset/white_dress/97e8f0a4745c9a89c8231d43f9ab213144ccb2bf.jpg  
      inflating: ./clothes_dataset/white_dress/985d17a4fb381c502dcb7d86a9276599b6786613.jpg  
      inflating: ./clothes_dataset/white_dress/98a0d4e2319240958041812f243e53b54e0aca68.jpg  
      inflating: ./clothes_dataset/white_dress/99247831f30261aa0e6c7ccf9cb3abf13c990f29.jpg  
      inflating: ./clothes_dataset/white_dress/99509e1f1a84da61bb8b0a8390c0b28b72fb1c86.jpg  
      inflating: ./clothes_dataset/white_dress/99c8d8bbb571b58c6fa94c8c2d0aaaf63b732489.jpg  
      inflating: ./clothes_dataset/white_dress/99da93f83ff7e5f2e26f90dd037034db4404849f.jpg  
      inflating: ./clothes_dataset/white_dress/9a08d8174c9cfd775d668f0655c296c07473edd2.jpg  
      inflating: ./clothes_dataset/white_dress/9a413d17fafdfdbbe83bd6fafe33f691e126515a.jpg  
      inflating: ./clothes_dataset/white_dress/9a52f9d9e73796126ddba6e21b0183c75dfabbb1.jpg  
      inflating: ./clothes_dataset/white_dress/9b645540be30131245d228a5c2b7f993a548ddfa.jpg  
      inflating: ./clothes_dataset/white_dress/9b87fe5128b5a6988c502d2332d49177af077814.jpg  
      inflating: ./clothes_dataset/white_dress/9bbaccdc0099d1ba5c114879210598624babcebd.jpg  
      inflating: ./clothes_dataset/white_dress/9be6aa5ec647f0e5209c474446d89d0e8cab5b4b.jpg  
      inflating: ./clothes_dataset/white_dress/9c06cc78eaa5110a887bdbb1801bcef07316e9cd.jpg  
      inflating: ./clothes_dataset/white_dress/9c4ce930c50a8037c1c156b763a6c4fd9fe27512.jpg  
      inflating: ./clothes_dataset/white_dress/9c726e0aae53013b7c3d76866a0a44be0a5ad947.jpg  
      inflating: ./clothes_dataset/white_dress/9cf521aef274e16e94d905cb0bbd2f89205a6f26.jpg  
      inflating: ./clothes_dataset/white_dress/9d0e486d3c638c73762909f830763f12afb13f4c.jpg  
      inflating: ./clothes_dataset/white_dress/9d777036e5df7a43aba547da8ebd6058047a54cc.jpg  
      inflating: ./clothes_dataset/white_dress/9d93a7077b8b0f4c81b5e99391fee7a80c71f6a6.jpg  
      inflating: ./clothes_dataset/white_dress/9ddfc9efb6a71241038aaa5f2453de6804bd8bbc.jpg  
      inflating: ./clothes_dataset/white_dress/9df7d4aebcc9f2db54f45d8374f8e16e8f16b92b.jpg  
      inflating: ./clothes_dataset/white_dress/9e0d7f8bb0cb4fe1c546769686df4f4d5908f3a9.jpg  
      inflating: ./clothes_dataset/white_dress/9f88f8656aa99dee6aba3345318853cce4f1d150.jpg  
      inflating: ./clothes_dataset/white_dress/9fbaba25cab3c621620699bff9e54ffe275fae38.jpg  
      inflating: ./clothes_dataset/white_dress/a0133551bf6dc0746f26437a18ba1e8e5b0738bf.jpg  
      inflating: ./clothes_dataset/white_dress/a06417895ce9a3792df152a4cbf9f8ad95b1f597.jpg  
      inflating: ./clothes_dataset/white_dress/a093e2a9355e596fb3d7312dc3b190e75662b7af.jpg  
      inflating: ./clothes_dataset/white_dress/a0a3ca84d13aaa7572b7aac9c9618adada4e7c25.jpg  
      inflating: ./clothes_dataset/white_dress/a11b4b67ecb1e341437ecfe120bd7b8a84be8f0d.jpg  
      inflating: ./clothes_dataset/white_dress/a11cfe888a78d55d9fdc0824dc24e917d0ae7a2b.jpg  
      inflating: ./clothes_dataset/white_dress/a1baefe3f52d9805c8995846b9062ed33c58f26f.jpg  
      inflating: ./clothes_dataset/white_dress/a2229850836842045e762ba5d9fbaef80939251d.jpg  
      inflating: ./clothes_dataset/white_dress/a2487c922644dcac0a25ce10775e005941771b0d.jpg  
      inflating: ./clothes_dataset/white_dress/a2b27c2a377d6c62ad19bb722da74ac58b8e9731.jpg  
      inflating: ./clothes_dataset/white_dress/a2dbec4a44fbc0b765f9bc16353af47e9a748788.jpg  
      inflating: ./clothes_dataset/white_dress/a3483a6561e96cd9a16ca204781d3a7bef77f1bc.jpg  
      inflating: ./clothes_dataset/white_dress/a35745c6d9e5c78eec083a765535de4a3d4bc802.jpg  
      inflating: ./clothes_dataset/white_dress/a3d4df22c800b31bd371b8b48e532522c02c2748.jpg  
      inflating: ./clothes_dataset/white_dress/a3ee899b295ee833f79e541ec6ea92ef4ff37da1.jpg  
      inflating: ./clothes_dataset/white_dress/a466ef094ec6cc19f649231ada1be8e26cc2902e.jpg  
      inflating: ./clothes_dataset/white_dress/a4774e327f5c3c8b209d921cdfabb593e8805bd4.jpg  
      inflating: ./clothes_dataset/white_dress/a480b704d2baffbfe7b0ac23ecdbc3083318bb79.jpg  
      inflating: ./clothes_dataset/white_dress/a483e13bedb7d9b34ec399c99d830ed0f8faaa54.jpg  
      inflating: ./clothes_dataset/white_dress/a4d944dd1e3939042a41171a1132a4ad6307b449.jpg  
      inflating: ./clothes_dataset/white_dress/a5563a1c25a4a4cd49909b8ccc3f6be65108e8fc.jpg  
      inflating: ./clothes_dataset/white_dress/a5d1f1ae14d6edbcf4888755ba897bf294328887.jpg  
      inflating: ./clothes_dataset/white_dress/a6738b2c3f1cae5bc6a371d9bee07e87d465b257.jpg  
      inflating: ./clothes_dataset/white_dress/a69ffc8e0b59fc845bb8d23fbc8a5a1c417b95e8.jpg  
      inflating: ./clothes_dataset/white_dress/a6a35576a0fafc2775a18866072c8c876820ad5d.jpg  
      inflating: ./clothes_dataset/white_dress/a6c0721e52d890cb27151a98a6e01299ca50bf12.jpg  
      inflating: ./clothes_dataset/white_dress/a6f38c3f17770211007aa870c10c6fc7ca6206ce.jpg  
      inflating: ./clothes_dataset/white_dress/a7284b8836c265ee1fe990885665721bf189a24a.jpg  
      inflating: ./clothes_dataset/white_dress/a72e824094fe4d52d8a926a139d30078e584253c.jpg  
      inflating: ./clothes_dataset/white_dress/a78060843d4c0c97d94df30a7f7e6d5a85daafcd.jpg  
      inflating: ./clothes_dataset/white_dress/a7f976461034dfa41a91c04a55deff0b78c9dd36.jpg  
      inflating: ./clothes_dataset/white_dress/a7fcba357c755597df894d06ab9fb89bbe417a80.jpg  
      inflating: ./clothes_dataset/white_dress/a886bf66be2124c33cbe22ca2373b5ee49f2c46a.jpg  
      inflating: ./clothes_dataset/white_dress/a88a3bc977876cfde24d39c0156c07c0f5f5138b.jpg  
      inflating: ./clothes_dataset/white_dress/a91f9bcc917fb39f8b9b70bb9eee38132a83bcea.jpg  
      inflating: ./clothes_dataset/white_dress/a93e5da437c17da5862b195403aad18bab2b162a.jpg  
      inflating: ./clothes_dataset/white_dress/a987f3569ba47c67e124d7298abf139c61f4f334.jpg  
      inflating: ./clothes_dataset/white_dress/a9a10834e0bed32a8bed164f46c69b2223826ae9.jpg  
      inflating: ./clothes_dataset/white_dress/a9bdad9203465ff95e946113ab776efd2fcb78ab.jpg  
      inflating: ./clothes_dataset/white_dress/aa045f995540d268bbf9ca1851e150d604b0f7bc.jpg  
      inflating: ./clothes_dataset/white_dress/aa07c5f8d4d3538b42c41e111facdbc730958d75.jpg  
      inflating: ./clothes_dataset/white_dress/aa0929b49cfa6439a9a10832601ab3d5b675f18b.jpg  
      inflating: ./clothes_dataset/white_dress/aa1e29f9a3b4cbf9f5296f094c953dda429b51d3.jpg  
      inflating: ./clothes_dataset/white_dress/aa596c5a23da6fb89f48494529bf20fb0eb6dc41.jpg  
      inflating: ./clothes_dataset/white_dress/aab207dab17007bb0ed1628626d8dced3ddbc8f0.jpg  
      inflating: ./clothes_dataset/white_dress/aafc3ab8d0aba29f23232c41167f353973e51410.jpg  
      inflating: ./clothes_dataset/white_dress/ab6ea2658005d0abc8a8503047846139c8e76c26.jpg  
      inflating: ./clothes_dataset/white_dress/ab782d71fe2bc0bad37236c7d7bc55ac66ca17e3.jpg  
      inflating: ./clothes_dataset/white_dress/aca5ef413ed03bce72061590c2daaa71d673094f.jpg  
      inflating: ./clothes_dataset/white_dress/accd2e76d1f9d06aed13be513f1fe212d6022324.jpg  
      inflating: ./clothes_dataset/white_dress/ad4a9f89f7da1599db86974d63740868e1ff63e9.jpg  
      inflating: ./clothes_dataset/white_dress/ad55b4e08b1d4fcf898140f940187ce158e2957d.jpg  
      inflating: ./clothes_dataset/white_dress/ad60edf003dba2aac07ce64bca87a4347302b8e9.jpg  
      inflating: ./clothes_dataset/white_dress/ad9d7eefcac4b75eea278d0cdd5ceb658fde615a.jpg  
      inflating: ./clothes_dataset/white_dress/ade8151fac0a6664e1f906691813dcdd2321eb3b.jpg  
      inflating: ./clothes_dataset/white_dress/aef0d9bf5e5aac6de48d5f01bc98246528cf55b4.jpg  
      inflating: ./clothes_dataset/white_dress/af554480c790353233cb1008a234fe58baf4a8a6.jpg  
      inflating: ./clothes_dataset/white_dress/af5c6532a8282e1af4353ccd8cdb33ffbfc621b0.jpg  
      inflating: ./clothes_dataset/white_dress/afa38c60fdfdf1e8da1956f72b0a02a8321cf3b3.jpg  
      inflating: ./clothes_dataset/white_dress/aff98505f209cd1ed59e2f1c33285b9b99726ca6.jpg  
      inflating: ./clothes_dataset/white_dress/b0536dce2d989265b9300d962efe785727d35199.jpg  
      inflating: ./clothes_dataset/white_dress/b0b55979f8f80a564ecad21934f8b341ce27454d.jpg  
      inflating: ./clothes_dataset/white_dress/b0cdbf2cf84a5f582daf75bf18558eacd9d3315e.jpg  
      inflating: ./clothes_dataset/white_dress/b1607d45c70c8f37dd14161b00065baa247b201f.jpg  
      inflating: ./clothes_dataset/white_dress/b1b88b353f15d2f5f7cb6ddfa48e1fbfb16038b0.jpg  
      inflating: ./clothes_dataset/white_dress/b24428bfd15898699cc5bafed0564ac5ae139fde.jpg  
      inflating: ./clothes_dataset/white_dress/b2469716140db07ae83160e163190e76066038fd.jpg  
      inflating: ./clothes_dataset/white_dress/b29f04ce433bf94c18739adcc5a458e5227e5986.jpg  
      inflating: ./clothes_dataset/white_dress/b2b0ba8faaaaafc50063b4dc5734a1994888ffd2.jpg  
      inflating: ./clothes_dataset/white_dress/b2d4f97e69642a98f8d46da98562335cd7740ffc.jpg  
      inflating: ./clothes_dataset/white_dress/b2dfb02a5f8151d829f4b29dd4dc72434c48f6ad.jpg  
      inflating: ./clothes_dataset/white_dress/b309d5b34b6f717773fd8c00a78e848a0ae78003.jpg  
      inflating: ./clothes_dataset/white_dress/b3275432afe5d3d43aa7913897884c9f684b4a76.jpg  
      inflating: ./clothes_dataset/white_dress/b329377007ca2195a3ead4812db80343cd5524c1.jpg  
      inflating: ./clothes_dataset/white_dress/b36d55da6ec74f2210b9d444129381f245ba19a6.jpg  
      inflating: ./clothes_dataset/white_dress/b3b19675b31658c57fcbe9cd9e91ee1e4685abf6.jpg  
      inflating: ./clothes_dataset/white_dress/b3ec52fd19aa05df3bf7a7b21dd427d45711e47e.jpg  
      inflating: ./clothes_dataset/white_dress/b41af414e10ff0ed8321b8ce483829db9cf02c43.jpg  
      inflating: ./clothes_dataset/white_dress/b43311a3132614883c7360e537b93e34f5b62b47.jpg  
      inflating: ./clothes_dataset/white_dress/b49a5123bb71c87c16a718e9f826d9dcb888fd56.jpg  
      inflating: ./clothes_dataset/white_dress/b5762e2746ba4007e10bebf68c4280958519ba49.jpg  
      inflating: ./clothes_dataset/white_dress/b5cf5835e1044ac3dd1038e307cf10aac0749269.jpg  
      inflating: ./clothes_dataset/white_dress/b610a864acb9109b4c15a46a118aafe89cd11a0a.jpg  
      inflating: ./clothes_dataset/white_dress/b61d957df96010118c0145faa2b08974770c09db.jpg  
      inflating: ./clothes_dataset/white_dress/b6b618490d326b52902b8384d2a3c959708f900d.jpg  
      inflating: ./clothes_dataset/white_dress/b7a0068debd91af522e4b37d2d7304fc26ff801b.jpg  
      inflating: ./clothes_dataset/white_dress/b819997bdac0f2bab17a087d00f0c32d7987ebab.jpg  
      inflating: ./clothes_dataset/white_dress/b83596a3e2e5fbd76bdb18de53689afc1287664e.jpg  
      inflating: ./clothes_dataset/white_dress/b8414a4ba5c710138ed18e82bb8ab988bbdd5ef0.jpg  
      inflating: ./clothes_dataset/white_dress/b8511c533a93a94673fcfcaaded551133f7ed651.jpg  
      inflating: ./clothes_dataset/white_dress/b86e11b24ab8ea1403043e117546996670f0627b.jpg  
      inflating: ./clothes_dataset/white_dress/b87158c563ba8a0de835467f9787fe2072d6c897.jpg  
      inflating: ./clothes_dataset/white_dress/b8c2883cd3256ca676c4cf908457bfdf6d9a9764.jpg  
      inflating: ./clothes_dataset/white_dress/b8d989ec2da7a58e97d21eacba17fcd04c976c71.jpg  
      inflating: ./clothes_dataset/white_dress/b91f51d4ba83f7d97b6e1abe8e18218482049d0c.jpg  
      inflating: ./clothes_dataset/white_dress/b9860951bd66cab21b260ad1dc13c3de81a8c4d9.jpg  
      inflating: ./clothes_dataset/white_dress/ba04045df2fd844b0f6044d6cb19a8543b1fdffc.jpg  
      inflating: ./clothes_dataset/white_dress/ba8c526e047d5d23169c91342d83a2e14f1f60ad.jpg  
      inflating: ./clothes_dataset/white_dress/baa2e4dc67558e7c8ee4bcc5064745c570621292.jpg  
      inflating: ./clothes_dataset/white_dress/bb3e7f69faf502228e7a7201822ae9badfb56e6f.jpg  
      inflating: ./clothes_dataset/white_dress/bb8bc6bccb3a06d8bb5b1c66bc8baf8153c91239.jpg  
      inflating: ./clothes_dataset/white_dress/bccc6ad34b6c32f7e27d27d0c14708fb4bb99bbf.jpg  
      inflating: ./clothes_dataset/white_dress/bcf7b73d0b3fb330bd002be2885cedbf4b6ef8f3.jpg  
      inflating: ./clothes_dataset/white_dress/bde3f4b5bea4177208f7f5fcaf683ce0244c666c.jpg  
      inflating: ./clothes_dataset/white_dress/bdf742e32e5baaa534eeb01277c8389fedd9f336.jpg  
      inflating: ./clothes_dataset/white_dress/be4561d1dd20607e3ad2d61403d357d047921e56.jpg  
      inflating: ./clothes_dataset/white_dress/bea60a057a1fc8c142c8a3cf23270b53eb52228c.jpg  
      inflating: ./clothes_dataset/white_dress/beca4e686f16f380d59f69e6a1eb8fd250e8b3df.jpg  
      inflating: ./clothes_dataset/white_dress/bf86c3ae1787fa66ef4af362d3461eb3b04f3bd3.jpg  
      inflating: ./clothes_dataset/white_dress/bfa279bc78c03df9e8910caebf3f4eeb7efc29de.jpg  
      inflating: ./clothes_dataset/white_dress/bff4ca8da21a11786a2b0c9f68930368184da810.jpg  
      inflating: ./clothes_dataset/white_dress/c12030c79b1c4f3f324b73abb975e708c873546a.jpg  
      inflating: ./clothes_dataset/white_dress/c12ecbedb78ce0fdc39cfeed777dbaebef03533f.jpg  
      inflating: ./clothes_dataset/white_dress/c13cacf4354267cfc12ee15d5250e99dbe4e2253.jpg  
      inflating: ./clothes_dataset/white_dress/c223cd6bbed53eec0b8cfa47201609ee7a2586d4.jpg  
      inflating: ./clothes_dataset/white_dress/c24e8ccc6499cee784c33e200ad5db8423cddfbc.jpg  
      inflating: ./clothes_dataset/white_dress/c272406a6c85cd5f4ab54a8489607f717a494606.jpg  
      inflating: ./clothes_dataset/white_dress/c2728298d85c7298aafc97f76e5c1844ebb9e7b3.jpg  
      inflating: ./clothes_dataset/white_dress/c2a7eabedf41cc3e2c1844af63275ed1a5a57e9e.jpg  
      inflating: ./clothes_dataset/white_dress/c2fb9ecb4a054305c7af49638aadc24098de8b64.jpg  
      inflating: ./clothes_dataset/white_dress/c38b3884dd5ae615512306a40c74ab013b0b37fa.jpg  
      inflating: ./clothes_dataset/white_dress/c3b57e457256ceb49d0b5c315b9ffcf1b27ae5e6.jpg  
      inflating: ./clothes_dataset/white_dress/c3eff6493cf41a0134453662133b29f8d62f3a6f.jpg  
      inflating: ./clothes_dataset/white_dress/c42734263e31958c1ae92acd7a25e6cd5aa1009a.jpg  
      inflating: ./clothes_dataset/white_dress/c46ed110745d94b5677b789aaec028aa830c34c8.jpg  
      inflating: ./clothes_dataset/white_dress/c49042ed272a6125bd0c85256ac270821ba41894.jpg  
      inflating: ./clothes_dataset/white_dress/c4c0be7bc2ebd512e8653bb7efa1b26d1b2f6765.jpg  
      inflating: ./clothes_dataset/white_dress/c4f0310315ae658582af55c66e4e7cf66a0def6e.jpg  
      inflating: ./clothes_dataset/white_dress/c5be03f567a335f089ddd8ed09c49d08838a6dcf.jpg  
      inflating: ./clothes_dataset/white_dress/c5c5fd09323ce71196e2503b9e7b83a55314cf2d.jpg  
      inflating: ./clothes_dataset/white_dress/c61f512b2fba12023ffebd939e9a56b58e0fe25d.jpg  
      inflating: ./clothes_dataset/white_dress/c638807db2b29fa2d4cb0bf8f257de174afb84f7.jpg  
      inflating: ./clothes_dataset/white_dress/c68ce1eb65dfd0db8eb0398f3faf42026b59dab2.jpg  
      inflating: ./clothes_dataset/white_dress/c6ca969ddcf31f84e44895e1460e49cee8616d6c.jpg  
      inflating: ./clothes_dataset/white_dress/c7337d69ae19fbe7456b76a624df16564acd6fab.jpg  
      inflating: ./clothes_dataset/white_dress/c7a8b6fb5b9e40272cfb2e2d6928c5644715a6c7.jpg  
      inflating: ./clothes_dataset/white_dress/c7baf2c34b2c13067dfd13f3d9781693d3176c2a.jpg  
      inflating: ./clothes_dataset/white_dress/c7bfd72946859d4d1bfbd34c8baa01ace898738e.jpg  
      inflating: ./clothes_dataset/white_dress/c7f2c77b4ce0efca4fc4a5eab28740ad8388e7d1.jpg  
      inflating: ./clothes_dataset/white_dress/c8137a921d1d04f00b0b5f5c91d3a6702611f25e.jpg  
      inflating: ./clothes_dataset/white_dress/c8298d79d4660aa3430c63f76cc3357dc3f47513.jpg  
      inflating: ./clothes_dataset/white_dress/c87f3905a17378321824379ce0a203c7d8aadaa6.jpg  
      inflating: ./clothes_dataset/white_dress/c8b9bfb5d9ccfe8d1834b02a7d04d4d1c8362d28.jpg  
      inflating: ./clothes_dataset/white_dress/c8ed8f790a7a866dc1c983fdee739cc1a6757cfd.jpg  
      inflating: ./clothes_dataset/white_dress/c9483ef51df2a9ccaabe4447e8e94297ca4f5b1c.jpg  
      inflating: ./clothes_dataset/white_dress/c961cbf62da38774cfb195aa7a14e3807609431b.jpg  
      inflating: ./clothes_dataset/white_dress/c988350ebfc684aca11a9603739863770d4777fd.jpg  
      inflating: ./clothes_dataset/white_dress/c9c0d740f0b7f4e4fa93026bf311ec7764abb358.jpg  
      inflating: ./clothes_dataset/white_dress/ca53c75b5afb279e50acc38b77cf05f588e2843c.jpg  
      inflating: ./clothes_dataset/white_dress/ca95f0118648dc4e28e767558f2c1f1ad1db424c.jpg  
      inflating: ./clothes_dataset/white_dress/cae9b5e50de76ef3aded9b9f547b01ce021cae3f.jpg  
      inflating: ./clothes_dataset/white_dress/caf19cb7bb812d20b62f8102aa0713faafb1e70b.jpg  
      inflating: ./clothes_dataset/white_dress/cb2b3d37a5424b7e47ad9b43e9801012ef3e4e0c.jpg  
      inflating: ./clothes_dataset/white_dress/cb4651b185be85714df26663012777269a44d6b1.jpg  
      inflating: ./clothes_dataset/white_dress/cbe588d80516edd89714f88762d9c7c4209d99c4.jpg  
      inflating: ./clothes_dataset/white_dress/cbf3094de8357db344a5cc721a0048d6a81e26d9.jpg  
      inflating: ./clothes_dataset/white_dress/cc0f621ea683ccd0e69e9daae37fde6b5eff85fa.jpg  
      inflating: ./clothes_dataset/white_dress/cc4fbe29202df2c5492913fcc664b3fd730933e6.jpg  
      inflating: ./clothes_dataset/white_dress/cc57ea37644886926b14b1eaeb90df31026562db.jpg  
      inflating: ./clothes_dataset/white_dress/ccc9948bc0e46da2780ce41388bd576dac382549.jpg  
      inflating: ./clothes_dataset/white_dress/cccec7e705ed1ebcbdd5805441aea451044e5e38.jpg  
      inflating: ./clothes_dataset/white_dress/ccd540662be3d8220f4a9a1f0fe014e1e2170df0.jpg  
      inflating: ./clothes_dataset/white_dress/cd3aee5cc100f0b5349c50860acfb729709f9484.jpg  
      inflating: ./clothes_dataset/white_dress/cd8015ebfed066097183cb1a738d1b3553f3c5e1.jpg  
      inflating: ./clothes_dataset/white_dress/cdfb0a1f3d3de4d870112adb79a8db8b9cd4c155.jpg  
      inflating: ./clothes_dataset/white_dress/ce029757004a4b069463cb717f972aa163438f1b.jpg  
      inflating: ./clothes_dataset/white_dress/ce13aa5a0ae91374b6192b4ccf1ba858c573e86e.jpg  
      inflating: ./clothes_dataset/white_dress/ce7a429f0a960720c405918eecef2f8962487d9a.jpg  
      inflating: ./clothes_dataset/white_dress/ce7d6d736a4af295dd12a2e504493862c03dccc3.jpg  
      inflating: ./clothes_dataset/white_dress/ce84827a71cbd3e61fb8e2cf0a0480a402a9ed31.jpg  
      inflating: ./clothes_dataset/white_dress/ce9b98a71cab4c2ad7cfc22388b7d4e3d201c0d7.jpg  
      inflating: ./clothes_dataset/white_dress/ced3155879d465b55e463504da857172040ce0a3.jpg  
      inflating: ./clothes_dataset/white_dress/cf3405251e7debc4e1cceb0e5e3307fcdc595d27.jpg  
      inflating: ./clothes_dataset/white_dress/cf96a87e717c4d7209f3f3acedef318b933518dd.jpg  
      inflating: ./clothes_dataset/white_dress/cfb6e115b26657cb9280a0f1c12f84a856a1675d.jpg  
      inflating: ./clothes_dataset/white_dress/cff058cd0bae03e26744579d8b185bd6c44fe9a5.jpg  
      inflating: ./clothes_dataset/white_dress/d01c59c6b4f2f20b8f7b82a55dff6dd4bc9fbe44.jpg  
      inflating: ./clothes_dataset/white_dress/d0b8c7513d439e3717b93cc826f40837cd741328.jpg  
      inflating: ./clothes_dataset/white_dress/d114229da485a04d3ad817f65b61e643cf80a6f6.jpg  
      inflating: ./clothes_dataset/white_dress/d12d97e8098b80408d85c676ba51110c5572367a.jpg  
      inflating: ./clothes_dataset/white_dress/d166478ed5ce11176b9cf908834e69330f1d13ec.jpg  
      inflating: ./clothes_dataset/white_dress/d23f739346eb3aa3d592bc37efa6df02e5a5cd09.jpg  
      inflating: ./clothes_dataset/white_dress/d313fe625452a5f45d693990e1c236e46e320a96.jpg  
      inflating: ./clothes_dataset/white_dress/d315133100dd136fc0d3e45c2e4eb6d6d04988e7.jpg  
      inflating: ./clothes_dataset/white_dress/d3b09d2e01079bc122b96ae2db8abe871164e1ec.jpg  
      inflating: ./clothes_dataset/white_dress/d503712eabf91133a46b08af826b7ada101e7f47.jpg  
      inflating: ./clothes_dataset/white_dress/d512ee63306e26432ed83df97d2aec9f4367e8d1.jpg  
      inflating: ./clothes_dataset/white_dress/d544be96ae5159ad621b272c7f54d47d607922db.jpg  
      inflating: ./clothes_dataset/white_dress/d58b9b44af5259330d36af0b2e2810ea956f237a.jpg  
      inflating: ./clothes_dataset/white_dress/d5b90690f5f8e31025c10a920865165a2b83e3c9.jpg  
      inflating: ./clothes_dataset/white_dress/d5cb4abe1df19a1d49d998efa312f177d3d34699.jpg  
      inflating: ./clothes_dataset/white_dress/d6b61a60952a27048e4e57631e26e6be02f77cdb.jpg  
      inflating: ./clothes_dataset/white_dress/d6c79cf6a0a4663822b4fee0e921153e6109a5af.jpg  
      inflating: ./clothes_dataset/white_dress/d6cdc95392f6d707db897ce08570d7adf6eb5f08.jpg  
      inflating: ./clothes_dataset/white_dress/d7cc75c3344b86dd18f6e7ea292cbcb80aee47fd.jpg  
      inflating: ./clothes_dataset/white_dress/d7ee126a618860f712c858ca7ac23ed4702b6df1.jpg  
      inflating: ./clothes_dataset/white_dress/d807df79a7cb42df35db662360c3a97968cdbf8b.jpg  
      inflating: ./clothes_dataset/white_dress/d83442a84da738cb8f15a749794b990bde4a42fc.jpg  
      inflating: ./clothes_dataset/white_dress/d91efcb4b581e70c5c2e686a3e7194ae6eb985ae.jpg  
      inflating: ./clothes_dataset/white_dress/d960cc5bb29bd0ccb8737228fa44e64886588ba3.jpg  
      inflating: ./clothes_dataset/white_dress/d9896ca0db0f461904f2665e720b59d26caaaed2.jpg  
      inflating: ./clothes_dataset/white_dress/da0a691eabb2f8f890462c66e2b98e3a2181605c.jpg  
      inflating: ./clothes_dataset/white_dress/da4dc1ae43d3d9dff2cfa32042bb211d8a2bcaef.jpg  
      inflating: ./clothes_dataset/white_dress/da58c8980ec0f6b62481e28fd1678996df7160a4.jpg  
      inflating: ./clothes_dataset/white_dress/da666435132554a907c01b49a119df9e8b0ad42a.jpg  
      inflating: ./clothes_dataset/white_dress/da75f4cff7b7a9b6a6e6589240054fda7260cded.jpg  
      inflating: ./clothes_dataset/white_dress/da9db342cb2e84780ec6013748b8c943228ea9fa.jpg  
      inflating: ./clothes_dataset/white_dress/dab1971ae0eb3210bd2ebc3baf925ce63ee414e5.jpg  
      inflating: ./clothes_dataset/white_dress/dab6557952e9f3a4a171989c04212909c4e4e8a4.jpg  
      inflating: ./clothes_dataset/white_dress/db6f20a839f7f7f273f692ab9a8a0f0bc9e5fd11.jpg  
      inflating: ./clothes_dataset/white_dress/dbe00a1b5b002c91339f0eca15fb04b030214603.jpg  
      inflating: ./clothes_dataset/white_dress/dc636ddd041c7ec0de1d18922a6e3d37a9c0e882.jpg  
      inflating: ./clothes_dataset/white_dress/dc8ef3ce7d4af07f444c5ab3c601053555eeb2cf.jpg  
      inflating: ./clothes_dataset/white_dress/dc9a3f60b7a0e3e02db0cae182493e562d722950.jpg  
      inflating: ./clothes_dataset/white_dress/dcb9e64f45dbf0476192377f2ab23a90a1d41ffa.jpg  
      inflating: ./clothes_dataset/white_dress/dd7dae1680637a58a2a1f06bd886bb54246918b7.jpg  
      inflating: ./clothes_dataset/white_dress/ddb8124681125001404f3b67ea3379ceea83c0ef.jpg  
      inflating: ./clothes_dataset/white_dress/ddbaaafa667ffe9e381394b8266d64316563b039.jpg  
      inflating: ./clothes_dataset/white_dress/ddda6d0ec9dba6ac6075b3d770a03879064ed7eb.jpg  
      inflating: ./clothes_dataset/white_dress/ddf7b7851ac1bfc2749c2da600b46f151ea996bd.jpg  
      inflating: ./clothes_dataset/white_dress/de0e8e4d5915be71afe43fa8225b153b1add7607.jpg  
      inflating: ./clothes_dataset/white_dress/de32f1417b93a38f79f5af87ffe1278691fef9ae.jpg  
      inflating: ./clothes_dataset/white_dress/de6471ee6ef1f3cf56bd83e6cd2ade9ee16a5eeb.jpg  
      inflating: ./clothes_dataset/white_dress/de7b87c354318bcb977166c675f958abdbabc6a9.jpg  
      inflating: ./clothes_dataset/white_dress/df0e891c987f46b2ffcc77f6225713a281791b36.jpg  
      inflating: ./clothes_dataset/white_dress/e04a05440b973f1796a0aaf74b20cf2b2be853d9.jpg  
      inflating: ./clothes_dataset/white_dress/e06d360a3003c575c3cf5aa5f2db056d06436b60.jpg  
      inflating: ./clothes_dataset/white_dress/e1a9aeebf095ea59e28b2df8aa2a8da3c18623cd.jpg  
      inflating: ./clothes_dataset/white_dress/e1b035451269f96096e1aa184b59d183e5ce7d41.jpg  
      inflating: ./clothes_dataset/white_dress/e202c59f677a0b08d99ca9a4c16264af6c4028e1.jpg  
      inflating: ./clothes_dataset/white_dress/e2415e046053d8418a8ed3a285632b56aadd0541.jpg  
      inflating: ./clothes_dataset/white_dress/e2416270c7c69de5255f7ed4c643a2e8f9503c5b.jpg  
      inflating: ./clothes_dataset/white_dress/e247656964ecec8d88dfb24a4c71bc8eabf6e527.jpg  
      inflating: ./clothes_dataset/white_dress/e2fc7cd1ff97b5a0c6b81cbbdc87033a1753882e.jpg  
      inflating: ./clothes_dataset/white_dress/e3321aa0e6f2f6db8bd7e31f17f50b1fb453007f.jpg  
      inflating: ./clothes_dataset/white_dress/e349fe0a93a492a7f45be155e13d2c7a09185e5f.jpg  
      inflating: ./clothes_dataset/white_dress/e3a778676fc99d5bd0453ce20cb190b433bf6bd7.jpg  
      inflating: ./clothes_dataset/white_dress/e3d34e4502ca65f0499fb7fa018ac1571656db1d.jpg  
      inflating: ./clothes_dataset/white_dress/e3f12601075f951012f1ccc99fdf07a93980fb65.jpg  
      inflating: ./clothes_dataset/white_dress/e4e889484f91c1ab405de3f094620010031fa797.jpg  
      inflating: ./clothes_dataset/white_dress/e5a6cca77b53fb0c5e01982009c0971da2b7d96b.jpg  
      inflating: ./clothes_dataset/white_dress/e5af46d649850de131c8843101f2ec869896901b.jpg  
      inflating: ./clothes_dataset/white_dress/e5d31ed83aaab4532708817a6cf210bb62f6228e.jpg  
      inflating: ./clothes_dataset/white_dress/e5d799d350c36e199b96ac565fc53d42583bca79.jpg  
      inflating: ./clothes_dataset/white_dress/e615b160cfeffa1f26d37dbe49c0a0a9606acde4.jpg  
      inflating: ./clothes_dataset/white_dress/e6474149d9982152bb92ef343c917ee914ec95a0.jpg  
      inflating: ./clothes_dataset/white_dress/e74a274c723a27b3442588c3ed2ba3050e32bdc5.jpg  
      inflating: ./clothes_dataset/white_dress/e78dfebb7a4dbdcc5b666de03fce3747d8fb654e.jpg  
      inflating: ./clothes_dataset/white_dress/e7a30e85fa26a6cabd29a7b8f57d7e835a377ca4.jpg  
      inflating: ./clothes_dataset/white_dress/e7ba1ae78e0e51f091aae8f86ff1a58fb2faa978.jpg  
      inflating: ./clothes_dataset/white_dress/e8726f8b23b92b6dc1f1a8adbcf1393153117103.jpg  
      inflating: ./clothes_dataset/white_dress/e897c8a2304ebfdc21c39d42591b081cce4d4d89.jpg  
      inflating: ./clothes_dataset/white_dress/e8a0215537fb89e5e73f28c39a8f99296391d771.jpg  
      inflating: ./clothes_dataset/white_dress/e8efa8efe1bde24e2338846c5c8d757c47652603.jpg  
      inflating: ./clothes_dataset/white_dress/e8f8469cfb6619d54e208399ac182cb5225f9674.jpg  
      inflating: ./clothes_dataset/white_dress/e912d3c7c5eafc4ceb737611c8c51f8f43efb3cd.jpg  
      inflating: ./clothes_dataset/white_dress/e9b43ba729b32c8a88fa03a70b9b7ab9286828be.jpg  
      inflating: ./clothes_dataset/white_dress/e9ee68cb9d8b29e74904e38c31f66c77f2983c4d.jpg  
      inflating: ./clothes_dataset/white_dress/ea35555717b932dc438b56d72d6590100a623e0c.jpg  
      inflating: ./clothes_dataset/white_dress/eb145e91bf8292dfa2020b0d786b2284fb864dbf.jpg  
      inflating: ./clothes_dataset/white_dress/ecc81b0dd04ea8e29d5126315667b54b4a4f3b4a.jpg  
      inflating: ./clothes_dataset/white_dress/ecefa3adcf1198ba7e1c8e7ae83e7ce13ae13d9f.jpg  
      inflating: ./clothes_dataset/white_dress/ed6192b46b976c31224259d07c8cda9d099aed66.jpg  
      inflating: ./clothes_dataset/white_dress/eddec33c49c77ad962c1936a6786801d48075fbf.jpg  
      inflating: ./clothes_dataset/white_dress/ee5b1d5fd9faf9deb3879fe69cb825c8cf2f6500.jpg  
      inflating: ./clothes_dataset/white_dress/ee97309781d5c3916046d7a4827bfcc6c3c17b3f.jpg  
      inflating: ./clothes_dataset/white_dress/eea78549428c0de4eb8a468f60e57ac95488aa81.jpg  
      inflating: ./clothes_dataset/white_dress/eecc7e045b84bc1ffaa6af0c6ea4db75852717e2.jpg  
      inflating: ./clothes_dataset/white_dress/ef170fd32880bad556f883e928380c64c4eea40f.jpg  
      inflating: ./clothes_dataset/white_dress/ef86bf5eee72dbe88f2d95f555fea7946ad098ca.jpg  
      inflating: ./clothes_dataset/white_dress/f03b7b70b69ad0feeb7d18a2928b13d7d1241acb.jpg  
      inflating: ./clothes_dataset/white_dress/f07f43e8056aed0df311b9ec1b03b94bec3a004f.jpg  
      inflating: ./clothes_dataset/white_dress/f09ea36ef7bafc16aef35025478c017320010dd1.jpg  
      inflating: ./clothes_dataset/white_dress/f0c51eeef530bff56e84f1a4139f8d107553e592.jpg  
      inflating: ./clothes_dataset/white_dress/f1b00eae3745b1f51e86d3d7fcc428e34b1e8c3d.jpg  
      inflating: ./clothes_dataset/white_dress/f20b5f40fd2bd66f99c385d72505c379fc0282cc.jpg  
      inflating: ./clothes_dataset/white_dress/f2cf59892bcc70c37221c208c6c2cd2cfc927c77.jpg  
      inflating: ./clothes_dataset/white_dress/f2fcff5afda5a459a1bec864303b996d05e1d940.jpg  
      inflating: ./clothes_dataset/white_dress/f317513f421e19a2dc8026b90e112a8d8fa0a188.jpg  
      inflating: ./clothes_dataset/white_dress/f32433a1bfc5df11eb4d3c8da1c2002142f38869.jpg  
      inflating: ./clothes_dataset/white_dress/f339175cb6cb0adbb63effa39a78b35b7ca219bf.jpg  
      inflating: ./clothes_dataset/white_dress/f36afa48570ee36aa1a1369f873a93829cbf2bac.jpg  
      inflating: ./clothes_dataset/white_dress/f37b5a8da25fa3b41d7c13bc580bd0f424a5d32d.jpg  
      inflating: ./clothes_dataset/white_dress/f3c7a15967617bbdfd92c6be48792dc7849e70e7.jpg  
      inflating: ./clothes_dataset/white_dress/f414efab8cf695392dcb55678a34517440876a4b.jpg  
      inflating: ./clothes_dataset/white_dress/f4597c5698b37c48cd2acaa85228461dc737898b.jpg  
      inflating: ./clothes_dataset/white_dress/f587c9dff3533481c8cb7641406368e642fea173.jpg  
      inflating: ./clothes_dataset/white_dress/f6817c4bec90a371eca7f0bec4cb357a281853e4.jpg  
      inflating: ./clothes_dataset/white_dress/f6820497bfd0da9eaeab67c93669edd7d9bff7c0.jpg  
      inflating: ./clothes_dataset/white_dress/f6a07af964cbe53a2978963e9b0ba67c8bd9d647.jpg  
      inflating: ./clothes_dataset/white_dress/f6d4c90ff9f0b6dc668b9d3227190fd787091ad1.jpg  
      inflating: ./clothes_dataset/white_dress/f6f47faac8aff371a540712565bd86a4f20207f3.jpg  
      inflating: ./clothes_dataset/white_dress/f703a73eec0a716a225f7c43c019da6bbcda632d.jpg  
      inflating: ./clothes_dataset/white_dress/f70b4a6451720ba9006575b8873848239dea00a6.jpg  
      inflating: ./clothes_dataset/white_dress/f7807fde77502cf9b8791bca17560c476d623f38.jpg  
      inflating: ./clothes_dataset/white_dress/f7a5a37c4f0d6363ac41134503d17150831eef10.jpg  
      inflating: ./clothes_dataset/white_dress/f8b014b6d580de84919d8639cbe86550fa0f646e.jpg  
      inflating: ./clothes_dataset/white_dress/f8e9437a10c9ae9fd747b46f2901331c0615ebfd.jpg  
      inflating: ./clothes_dataset/white_dress/f90ebede619c628201ae42dd29a183a50c37c0ea.jpg  
      inflating: ./clothes_dataset/white_dress/f924493ec6fcadee331efc468ebd78dd25dbab19.jpg  
      inflating: ./clothes_dataset/white_dress/f9605540025cbe3a068b47203ca44d2414a1f0ca.jpg  
      inflating: ./clothes_dataset/white_dress/f9c49bab6f48313b64f74f5d6ffcac499f4061ec.jpg  
      inflating: ./clothes_dataset/white_dress/f9d870a0ca87a64141fd4a5cdbe9df1d9fb6dd22.jpg  
      inflating: ./clothes_dataset/white_dress/f9ee4777a65e6cc265999fd8b2a18277cf765553.jpg  
      inflating: ./clothes_dataset/white_dress/fa1ffe5cae87d4c369000dfe46fcc83340058a2e.jpg  
      inflating: ./clothes_dataset/white_dress/fa7ab0f3b740026e84c86dd0b4a9c25f38c8b944.jpg  
      inflating: ./clothes_dataset/white_dress/fa7cc1b801dc355b5f4590a6e579cdb0a211a686.jpg  
      inflating: ./clothes_dataset/white_dress/fa98b6bdcdd050c53abf3079eded3846e9f08b64.jpg  
      inflating: ./clothes_dataset/white_dress/fad727c549b4c8f14119555217cbeaba4dd4a968.jpg  
      inflating: ./clothes_dataset/white_dress/fb948d9d4d940c37e6d0069891bf5a867a65b7f6.jpg  
      inflating: ./clothes_dataset/white_dress/fba8379594931db4ae71cf52d7d26022a6304862.jpg  
      inflating: ./clothes_dataset/white_dress/fbd43afc8b69cc3e56099578bca69d1a2fa70c1c.jpg  
      inflating: ./clothes_dataset/white_dress/fbeecc68525f6ac3a6f24f975c8a9f2701a45a9a.jpg  
      inflating: ./clothes_dataset/white_dress/fbf901976c50b3822b8fc02e232b8c650e1dea93.jpg  
      inflating: ./clothes_dataset/white_dress/fbfe05587b766efe646d0ed02ae4443009668500.jpg  
      inflating: ./clothes_dataset/white_dress/fc8ec095d6c8f2ec463575d4aed3165782625238.jpg  
      inflating: ./clothes_dataset/white_dress/fd085e7f8b43449b1be41bfea84982f887224e11.jpg  
      inflating: ./clothes_dataset/white_dress/fd3990229e912e47099b1a7311d707a0070e1217.jpg  
      inflating: ./clothes_dataset/white_dress/fd6205cd9b275ec9a852f693d38beaf141f5698d.jpg  
      inflating: ./clothes_dataset/white_dress/fd763aef5904ceb0206eacbf0e84975c54fe9eb3.jpg  
      inflating: ./clothes_dataset/white_dress/fd95f53e7937f816cf495e7b52da425ce81fd781.jpg  
      inflating: ./clothes_dataset/white_dress/fdadf1456d09f3c85dd94038ad3174049ee75fd2.jpg  
      inflating: ./clothes_dataset/white_dress/fe806e8c580cc183b2d9cd604eac01cb3be7e43a.jpg  
      inflating: ./clothes_dataset/white_dress/fe9c980d3f58b794aef767eb656f9676a1b62475.jpg  
      inflating: ./clothes_dataset/white_dress/feb36d41dcef0cc6b433fea772c4faac2db32b6b.jpg  
      inflating: ./clothes_dataset/white_dress/fec343b87e2bc6aaf230cc704e527e788d430197.jpg  
      inflating: ./clothes_dataset/white_dress/fed237044dc7b96ef4f9b404bbb0b8ba75c93b00.jpg  
      inflating: ./clothes_dataset/white_dress/ff0243162c763b6b54239ed09f82043d43c78aca.jpg  
      inflating: ./clothes_dataset/white_dress/ff41cc63436b9de5b2567e6a5fcc81af4d3dad77.jpg  
      inflating: ./clothes_dataset/white_dress/ffcf0042f6d895c6681d57e75382030a65c53061.jpg  
      inflating: ./clothes_dataset/white_dress/fffd3b22d82bf72353db0fc022ef40678f7787bc.jpg  
      inflating: ./clothes_dataset/white_pants/005db4f023a438e3cde9c9c3a5abe5550b95ed90.jpg  
      inflating: ./clothes_dataset/white_pants/00b2234fa8d634a80cce7f9bf3adf62ac2cf95d8.jpg  
      inflating: ./clothes_dataset/white_pants/03ec14a1a823f13cd3527ab2eeb87d02520b388e.jpg  
      inflating: ./clothes_dataset/white_pants/055bec6ce54d5ef5bffb58db72f68b9d9508e843.jpg  
      inflating: ./clothes_dataset/white_pants/059771be080428a8928792ce953b1bf0d243e917.jpg  
      inflating: ./clothes_dataset/white_pants/06cd77e0864fd62667eb0fa4e204f33d52cbf2a3.jpg  
      inflating: ./clothes_dataset/white_pants/094b85788a4f7d29c5c710321b9fa79e1b9a7cb4.jpg  
      inflating: ./clothes_dataset/white_pants/09d56d3e4855c296bd397a885d19351b8d215c5c.jpg  
      inflating: ./clothes_dataset/white_pants/0d16245b52a63449e857b78f192171ce74e72e06.jpg  
      inflating: ./clothes_dataset/white_pants/0d2a340154236ef00f6061a0d154d31a14894654.jpg  
      inflating: ./clothes_dataset/white_pants/0de798a346560ea3a78c5baff67fe606b611f023.jpg  
      inflating: ./clothes_dataset/white_pants/0e117415daa3886c981b0052e9b7cd2ba2e421e2.jpg  
      inflating: ./clothes_dataset/white_pants/0f3d89776d6116d4001070bedf46e418fba88732.jpg  
      inflating: ./clothes_dataset/white_pants/11aab9dcb1ef9c470d7a4d47a695a30363553213.jpg  
      inflating: ./clothes_dataset/white_pants/11ab72cb7b1df244a03122d5aecda827d99b5937.jpg  
      inflating: ./clothes_dataset/white_pants/12ad5eecd3f960e4abce5e153a1f73485c1faa33.jpg  
      inflating: ./clothes_dataset/white_pants/132ecd2c2a147f92aa2f72d3ccccd19f0ef042ca.jpg  
      inflating: ./clothes_dataset/white_pants/13f6e6e2902c0eef3460e4feb4baffbdf2246bc0.jpg  
      inflating: ./clothes_dataset/white_pants/14a8f923a40a95480513e3f433261097f5aeb8ee.jpg  
      inflating: ./clothes_dataset/white_pants/157a09a12713da8c27fcf25168b6253eaece4e84.jpg  
      inflating: ./clothes_dataset/white_pants/15b58da191637ff7a6d6b476318e7998a073897c.jpg  
      inflating: ./clothes_dataset/white_pants/16c8989572b6ebd6193cd585bec54df696a564b2.jpg  
      inflating: ./clothes_dataset/white_pants/1a9bcbc3c0d5b470db790e01febfb4ba7e409bbf.jpg  
      inflating: ./clothes_dataset/white_pants/1b27a45d30e36736e5a85add10a4d2d2f5dbf274.jpg  
      inflating: ./clothes_dataset/white_pants/1bcbe54dc7a40f41b0107f08f00af1d30527f94a.jpg  
      inflating: ./clothes_dataset/white_pants/1d406670a6681d0929044f583bbcfa17e6ffa1e3.jpg  
      inflating: ./clothes_dataset/white_pants/1d91cc2074628798bb562f86106ae718fe524ace.jpg  
      inflating: ./clothes_dataset/white_pants/1dd5b9ba2d08f5cdd5613b37a8a9321610e29472.jpg  
      inflating: ./clothes_dataset/white_pants/1e53d2a36918eedf09f0956d151c54faacbf5ab5.jpg  
      inflating: ./clothes_dataset/white_pants/1f882e84c1a293e3001fd40d977c56480f5be052.jpg  
      inflating: ./clothes_dataset/white_pants/21a0ebd678695ad4bacf6831d93ac0103832309d.jpg  
      inflating: ./clothes_dataset/white_pants/21a3eb5a2f2e9a1729c2369ab4f36763002bd278.jpg  
      inflating: ./clothes_dataset/white_pants/21b402fb9d95e1d069c71859e8d4298ecf691799.jpg  
      inflating: ./clothes_dataset/white_pants/21eae466d5f4c831e077350ed210048f5a55ce53.jpg  
      inflating: ./clothes_dataset/white_pants/2355647b75f6c0066d80b217086cf7ad6add1209.jpg  
      inflating: ./clothes_dataset/white_pants/241344311e2a1a7300b04f30c55ba20b8d24dc44.jpg  
      inflating: ./clothes_dataset/white_pants/24b2312716af3e4bbd2b6fe7280132b85a356ca8.jpg  
      inflating: ./clothes_dataset/white_pants/25ce6bd30337290584a097fb490f15604bb666da.jpg  
      inflating: ./clothes_dataset/white_pants/26f2c950885b98359fa295d4077db811a815e205.jpg  
      inflating: ./clothes_dataset/white_pants/26f45a032865ebcd3b5ea24f3b6f3123cf6cabd0.jpg  
      inflating: ./clothes_dataset/white_pants/27a295dab939ede6b68f60a9b554cc41ef45a658.jpg  
      inflating: ./clothes_dataset/white_pants/28b4df04eb16415773c5b66a6b5b05b34dd8cc02.jpg  
      inflating: ./clothes_dataset/white_pants/29119768e9694d862be0cdd5655c7bf9fc7ae8a6.jpg  
      inflating: ./clothes_dataset/white_pants/2a10313b17c79b313deb0d033060ce564e287570.jpg  
      inflating: ./clothes_dataset/white_pants/2a29fbe2d3b3400864e156d405d9432bf8f248c6.jpg  
      inflating: ./clothes_dataset/white_pants/2b9207cbd771f56c024115da9c987fdf42926b5d.jpg  
      inflating: ./clothes_dataset/white_pants/2b94704b33d1185fab7f928acb0b6ef1e5597e91.jpg  
      inflating: ./clothes_dataset/white_pants/2c604c141602c0ea1ae2eb1f2abb11dca666a660.jpg  
      inflating: ./clothes_dataset/white_pants/2e2055908980d1218455e8d64fa4f9718f5d14fc.jpg  
      inflating: ./clothes_dataset/white_pants/2fafa1cb6de59c3a0b29b2499d74075b39988330.jpg  
      inflating: ./clothes_dataset/white_pants/3019646b6d4c0821e5cd041366a8fee6649bb5c8.jpg  
      inflating: ./clothes_dataset/white_pants/31a16863ff0e5cf38560e0234f30b1dc9f550e33.jpg  
      inflating: ./clothes_dataset/white_pants/31b2609cda415c9ea1e41e2d2a10977834ab495c.jpg  
      inflating: ./clothes_dataset/white_pants/31f51104af794156c50cbb5810d15fe361c747e1.jpg  
      inflating: ./clothes_dataset/white_pants/34598f75723207f25f4c4c0782f1aadbbf130feb.jpg  
      inflating: ./clothes_dataset/white_pants/36cebdf1f001b72417f6d0de077cef82ff44777f.jpg  
      inflating: ./clothes_dataset/white_pants/3774a7158fe2f55e19df9e198af28d97038ae0b3.jpg  
      inflating: ./clothes_dataset/white_pants/38c8a022f4a42e18010a0cf3f1e9c337fea855cb.jpg  
      inflating: ./clothes_dataset/white_pants/39f8cb118f65a0a481096fa629072575861aa8b8.jpg  
      inflating: ./clothes_dataset/white_pants/3ad383731c09565edd302cc6a6488017589a84f5.jpg  
      inflating: ./clothes_dataset/white_pants/3baef2a1547dc9ea887ae52521d078b138e2d21a.jpg  
      inflating: ./clothes_dataset/white_pants/3cab763276e2ed62b330bb62bbea3c61bec48a03.jpg  
      inflating: ./clothes_dataset/white_pants/3eab08fa8b24d448029d462d204d9f8d6b63af6d.jpg  
      inflating: ./clothes_dataset/white_pants/3f057737f32614912d8347d3a9bddea69cb7aad3.jpg  
      inflating: ./clothes_dataset/white_pants/3fb2464a337c596fb80a3b3252cfbef3d6fbd864.jpg  
      inflating: ./clothes_dataset/white_pants/4188bad1a9091313f09cd11a1e4b62be3309f20e.jpg  
      inflating: ./clothes_dataset/white_pants/43751b75c9c2263a7a0512029f71231ec6038c9d.jpg  
      inflating: ./clothes_dataset/white_pants/43f35d3cc464cf745fdee798b0a1f9035afc1ecb.jpg  
      inflating: ./clothes_dataset/white_pants/44a62956b3ef2d4803b8c7a245432ef63fd5aa94.jpg  
      inflating: ./clothes_dataset/white_pants/48475d21aa1a14030b978b835077f3847c23018b.jpg  
      inflating: ./clothes_dataset/white_pants/49122cee9352356863483ef16bbc9856f757efdb.jpg  
      inflating: ./clothes_dataset/white_pants/494c470c9537c966fbace904c04058b5a8111e0a.jpg  
      inflating: ./clothes_dataset/white_pants/497b72e2dedfc4546316bb24083cbfc713de81e8.jpg  
      inflating: ./clothes_dataset/white_pants/498c472f939290ff8cd0bf708d9d3218dad02f9e.jpg  
      inflating: ./clothes_dataset/white_pants/4a27f20cf6d9811aab4e4bc12ccec482b61fa94e.jpg  
      inflating: ./clothes_dataset/white_pants/4a5a32ec12d090ea1c95e2906fe05ca84884d265.jpg  
      inflating: ./clothes_dataset/white_pants/4a820c2ac5594baeca28e2d7e4bef20d2bbccbb8.jpg  
      inflating: ./clothes_dataset/white_pants/4c2fffdb0134af0571bcf5a7034babbd9e0ec575.jpg  
      inflating: ./clothes_dataset/white_pants/4cd88d9365f72efe2b819ac81bcb49ff7433bd46.jpg  
      inflating: ./clothes_dataset/white_pants/50a6fc3f80612c5e2712f2f2ba0d050ce11fdec2.jpg  
      inflating: ./clothes_dataset/white_pants/517cbb746eb9ea38aae1d491dd6d7f0067c8c8b8.jpg  
      inflating: ./clothes_dataset/white_pants/52a7231ba646abf9fa4f4797f5114ebc99bacd31.jpg  
      inflating: ./clothes_dataset/white_pants/52bcc67bbcbcdd9c5732b26593090aa7dc8d2102.jpg  
      inflating: ./clothes_dataset/white_pants/550a822f268b6c31e14f51798d57525d22b21d1a.jpg  
      inflating: ./clothes_dataset/white_pants/552b7778a2f713db903efcf0efe6ae1c18d0ba68.jpg  
      inflating: ./clothes_dataset/white_pants/56e30926bb460dda90ab0401833e4e3d3b2594ae.jpg  
      inflating: ./clothes_dataset/white_pants/5791702ce3bc0b7096946c890d8c804e716a6ccb.jpg  
      inflating: ./clothes_dataset/white_pants/57f6a2e0ba06df835a42207aa39df9bc3e4179dd.jpg  
      inflating: ./clothes_dataset/white_pants/5841dab2031316c0394668f4b1e43368e57f5560.jpg  
      inflating: ./clothes_dataset/white_pants/59e3989478c86fba5e76db4ebb54b3b0177a396a.jpg  
      inflating: ./clothes_dataset/white_pants/5a42a654b3a5767e9b1b0072d9e43e5996752e82.jpg  
      inflating: ./clothes_dataset/white_pants/5c877b29ab916855f214e2baf8c51d37e571156e.jpg  
      inflating: ./clothes_dataset/white_pants/5fbee223975d7e0baf8a7cfc1ddf9c093155ac58.jpg  
      inflating: ./clothes_dataset/white_pants/615c16a26313f91533972c992210f524adea0089.jpg  
      inflating: ./clothes_dataset/white_pants/636f55be1758ef2c9ad13c8b78f7d38e28216eda.jpg  
      inflating: ./clothes_dataset/white_pants/643982ec8f40e322898bd1a810ff6e66fd1ff704.jpg  
      inflating: ./clothes_dataset/white_pants/64b0f3404606f83c9a19c8f48ef57fce54ee33c4.jpg  
      inflating: ./clothes_dataset/white_pants/64cf6d011389d70f37d278baa75f95ddf2651c5d.jpg  
      inflating: ./clothes_dataset/white_pants/655b2cc0378c4f4bc2e99409c16b1e4d25ebe8cb.jpg  
      inflating: ./clothes_dataset/white_pants/6606c70e547a1961790f922a394e381d444223be.jpg  
      inflating: ./clothes_dataset/white_pants/66da71317c44d8ee4b1bb6cb7a0b377a4fc7e0b5.jpg  
      inflating: ./clothes_dataset/white_pants/67acc8957e8ea5597cf2abec7adf3a32608a6aa2.jpg  
      inflating: ./clothes_dataset/white_pants/6856a4b9ea61b8acae833b6eceea2f4617037a82.jpg  
      inflating: ./clothes_dataset/white_pants/68fbd17aa53aa0de809365d0812f520201f57c08.jpg  
      inflating: ./clothes_dataset/white_pants/6b52444ec6a59112d7ac4222f66768e844d3fd56.jpg  
      inflating: ./clothes_dataset/white_pants/6ec35cfc2a72ebdeb6a12e91c2e471c9b75c0c9f.jpg  
      inflating: ./clothes_dataset/white_pants/70bb7e691abae762f77b0380332cb5868851b1a6.jpg  
      inflating: ./clothes_dataset/white_pants/71e1fb87d0f88b6020582fdfe890a28925347c93.jpg  
      inflating: ./clothes_dataset/white_pants/73ea02a590f0533069e42787cb339a5af8882876.jpg  
      inflating: ./clothes_dataset/white_pants/7446ebc83472e8c07811035508144eab68f916fc.jpg  
      inflating: ./clothes_dataset/white_pants/74750a2902578679ba539547bbbf59882d83b179.jpg  
      inflating: ./clothes_dataset/white_pants/75aab0a54a8cffff2ab23a080187a34bf3bd1ad4.jpg  
      inflating: ./clothes_dataset/white_pants/75c96877fa19468eca2f585492da9fdc59026489.jpg  
      inflating: ./clothes_dataset/white_pants/77fb431b5369c598a015de162eb267b53e70eb84.jpg  
      inflating: ./clothes_dataset/white_pants/78214000258653dcd2f4e2b334ea4c21d1be00f9.jpg  
      inflating: ./clothes_dataset/white_pants/79440bbb1fb113bff5e831de4a60c48b6cf8220a.jpg  
      inflating: ./clothes_dataset/white_pants/7b7cc4dba24bd4b2854fba430b3c77c15976a567.jpg  
      inflating: ./clothes_dataset/white_pants/7bc2cb2f33142048605856c8ceb3c2bc0da0f41e.jpg  
      inflating: ./clothes_dataset/white_pants/7bc7bf784204e47a33bffa721a6cc2d30cf1d25f.jpg  
      inflating: ./clothes_dataset/white_pants/7c2d5768aa5e6990a2375c4e95332fca98719bd7.jpg  
      inflating: ./clothes_dataset/white_pants/7c4b9c688af15f26f502d239848419a0768cd748.jpg  
      inflating: ./clothes_dataset/white_pants/7c7909cf211b3ae04e2c9ea6529ac7bcdc2f36c1.jpg  
      inflating: ./clothes_dataset/white_pants/7c81af01250c16cc9d9c07b14ff7058f5ddbd8ff.jpg  
      inflating: ./clothes_dataset/white_pants/7da86f97aafa712048795405ce015a11a170dc4a.jpg  
      inflating: ./clothes_dataset/white_pants/7f45f00a13a3bdc5574cc34762beb123c719ac2e.jpg  
      inflating: ./clothes_dataset/white_pants/7f8d2487938f82e7baab2f5051e8e69ccace4632.jpg  
      inflating: ./clothes_dataset/white_pants/7fa0e10ec21b59f75ac77751be3e03053394a135.jpg  
      inflating: ./clothes_dataset/white_pants/8008017b4ee1473328b461f766fdd408269e2081.jpg  
      inflating: ./clothes_dataset/white_pants/8097ab4fa9f27fe10134fde144e948a1ef3b4f40.jpg  
      inflating: ./clothes_dataset/white_pants/819be99035ae8c340511e04e3049ffac6078e8c9.jpg  
      inflating: ./clothes_dataset/white_pants/81eefe17d094646861d831e026351ade95719cc7.jpg  
      inflating: ./clothes_dataset/white_pants/825954dbab024ffc8997721cf65138fffe06a1f3.jpg  
      inflating: ./clothes_dataset/white_pants/830181ee932c788960a03a45fc38e18fc1f3d6ab.jpg  
      inflating: ./clothes_dataset/white_pants/83df8f6107fd5a740a58b7b7a6883e962f610a5a.jpg  
      inflating: ./clothes_dataset/white_pants/83fa8c337d9a32baf76ba01805f9e47d0b1bf4f9.jpg  
      inflating: ./clothes_dataset/white_pants/8458e3e5cafdbe04e03528c3a6837938709b6b19.jpg  
      inflating: ./clothes_dataset/white_pants/857947f044a56e239331aa278bad9f9a1d166c9f.jpg  
      inflating: ./clothes_dataset/white_pants/8650dfe2b32eea2ab8d30dd4057cb22e0d7a0396.jpg  
      inflating: ./clothes_dataset/white_pants/882cac789599d14884ca5c6018991bf4b8485045.jpg  
      inflating: ./clothes_dataset/white_pants/8886e1c0d8a04d907c1ff2edeaa41797e5289f81.jpg  
      inflating: ./clothes_dataset/white_pants/89b064dd7f1823887af2c31c2b081d048a4b214d.jpg  
      inflating: ./clothes_dataset/white_pants/8b3319b1d4f3cce60f1a084ec11a715514798601.jpg  
      inflating: ./clothes_dataset/white_pants/8bcc525a2521ee6308407ff2341dafe5433aeec5.jpg  
      inflating: ./clothes_dataset/white_pants/8c9b33058a21237bb4ccab514fb6c842c0b61f88.jpg  
      inflating: ./clothes_dataset/white_pants/8d71c566b12e28425f82958249e2eaf6a06c8104.jpg  
      inflating: ./clothes_dataset/white_pants/8da3f6df62a93284c414fa0d922a11db3bcc2d8a.jpg  
      inflating: ./clothes_dataset/white_pants/8e75a963a94c8cda7eda41ee15470fc04408e5f3.jpg  
      inflating: ./clothes_dataset/white_pants/8f0d5bd9391a0714da6e4e84be5cfa5c3ce1e806.jpg  
      inflating: ./clothes_dataset/white_pants/8fe4df4512acbc4db2e2dcd9f11659828722372c.jpg  
      inflating: ./clothes_dataset/white_pants/90e22fcc99a590f80a84ca04f17c052b0372ba58.jpg  
      inflating: ./clothes_dataset/white_pants/91998ccc93ea2f6ebdbb2cb880578a17a35f40bd.jpg  
      inflating: ./clothes_dataset/white_pants/91fb0b47ac694f6450d0062c44480170c138d5aa.jpg  
      inflating: ./clothes_dataset/white_pants/92366a5e71bc5f6accf82d05749e900739217573.jpg  
      inflating: ./clothes_dataset/white_pants/93cb2d1ddf0a674751d972ac05c4df6c6d9e2556.jpg  
      inflating: ./clothes_dataset/white_pants/989df562492f19f36f56d7548c49ef63e2172d95.jpg  
      inflating: ./clothes_dataset/white_pants/98de034be9e9a60e7b88068f3f2657cfe19e96a2.jpg  
      inflating: ./clothes_dataset/white_pants/9961f37218519b2097e140708d5f1d34b1ca581c.jpg  
      inflating: ./clothes_dataset/white_pants/9a6847d1ddbb367df02536f00a65be6e85547f1d.jpg  
      inflating: ./clothes_dataset/white_pants/9aad4f3e272617234e40820eb612754b9a4e4402.jpg  
      inflating: ./clothes_dataset/white_pants/9bb8632bfcd3e73c0cd4c03556733a26e9ff7f55.jpg  
      inflating: ./clothes_dataset/white_pants/9bbd8e7f8bcb42d0b137614b9af2fc6d35b0378b.jpg  
      inflating: ./clothes_dataset/white_pants/9ea28f75339c742f283b71300e83cee75988b98c.jpg  
      inflating: ./clothes_dataset/white_pants/9f5473a1ebb879716437078b0dc2a09b5aedc5d8.jpg  
      inflating: ./clothes_dataset/white_pants/a026e396ff792997959c5fcceba7e6fa4dc14b8a.jpg  
      inflating: ./clothes_dataset/white_pants/a0e5c5fb28899dd593a78b2535df80fd963b878c.jpg  
      inflating: ./clothes_dataset/white_pants/a19e6a736f56903e3adf3ca72cce9906ee954a45.jpg  
      inflating: ./clothes_dataset/white_pants/a1afde9fb308dae5ef3b930da8d4b0e4e5751020.jpg  
      inflating: ./clothes_dataset/white_pants/a1e4262ca93f8b21a808df71a0ab09fb27aa49a3.jpg  
      inflating: ./clothes_dataset/white_pants/a219997a0d1c37d88dbf1d75bfaa016fb8e6a4aa.jpg  
      inflating: ./clothes_dataset/white_pants/a2a1ffc802d2a1a2967d2d476c42305536f8e191.jpg  
      inflating: ./clothes_dataset/white_pants/a36c5281893c4885789ea4a3a6d1a1e828530b03.jpg  
      inflating: ./clothes_dataset/white_pants/a43961dd57a34bf4b0930efabc4358ccbcc4759c.jpg  
      inflating: ./clothes_dataset/white_pants/a55fea190ae034f906c4396e2f6ff69241d245dc.jpg  
      inflating: ./clothes_dataset/white_pants/a5a0212b112bffe3129769299aa2aa07893e9b59.jpg  
      inflating: ./clothes_dataset/white_pants/a5d1249f5e4a09280fcabc7d9d62520a690df9c0.jpg  
      inflating: ./clothes_dataset/white_pants/a5e3742be1d1fa510506a754168206aea5203ea2.jpg  
      inflating: ./clothes_dataset/white_pants/a6c6f7bed98032a773753b8ba0e07aa14a60985c.jpg  
      inflating: ./clothes_dataset/white_pants/a71261855008473a184b21fadb4d0b8b588eed8d.jpg  
      inflating: ./clothes_dataset/white_pants/a75cbe0ca7f1b50c79077065c0539bdf19e0447e.jpg  
      inflating: ./clothes_dataset/white_pants/a77a750961a50aa267274ca82a7bc1865fe0258e.jpg  
      inflating: ./clothes_dataset/white_pants/a7d12b006dade8273a8805182d4762e330bc8223.jpg  
      inflating: ./clothes_dataset/white_pants/aa6767c7d924eede090c665482907d2bf24e5a3c.jpg  
      inflating: ./clothes_dataset/white_pants/ab631cce0d56793f16796e5c6054bec2883640b9.jpg  
      inflating: ./clothes_dataset/white_pants/ab9409af4bd9feae9bd05660ba0cbd96c7ee94f8.jpg  
      inflating: ./clothes_dataset/white_pants/ab9ddf91259b35bf4710c78a1e409687abcbc02a.jpg  
      inflating: ./clothes_dataset/white_pants/acd03af63deb061b41750b6b9fe6c0c2f1e57344.jpg  
      inflating: ./clothes_dataset/white_pants/ad34c444839584cc2d2f9332f36ce65a3f7c61a0.jpg  
      inflating: ./clothes_dataset/white_pants/ad515a9b36c894a2f72eaa21ab77a945c3be97ab.jpg  
      inflating: ./clothes_dataset/white_pants/adb4f16a5df1c463a00019051607f2f1b8371406.jpg  
      inflating: ./clothes_dataset/white_pants/ae3f22c170437fd61f9437533c8cdf6a7b7b5aec.jpg  
      inflating: ./clothes_dataset/white_pants/afe984edf08864c22f10513837d4793062a06519.jpg  
      inflating: ./clothes_dataset/white_pants/b05a28b88dc7fe20b6be75150785d04a36129601.jpg  
      inflating: ./clothes_dataset/white_pants/b0f84bac59fb34108737c1f1a6a086fad0a1b781.jpg  
      inflating: ./clothes_dataset/white_pants/b1c44f27329462a4edd9ebcb9f4380af1f5b06e0.jpg  
      inflating: ./clothes_dataset/white_pants/b33d97e48bc8020601726420914b3e90beb5fa71.jpg  
      inflating: ./clothes_dataset/white_pants/b36d5134d3883bf5d9899f252688181245ef0388.jpg  
      inflating: ./clothes_dataset/white_pants/bab9c07582ddd3dd53a8344bcc773c467e58f4f6.jpg  
      inflating: ./clothes_dataset/white_pants/bac434d6c48f8d53f460db449d673e455d63e944.jpg  
      inflating: ./clothes_dataset/white_pants/bbf4d74c2af1afb50a52b8dec6999f230e3af4d9.jpg  
      inflating: ./clothes_dataset/white_pants/bc2809a6e153b0d99405ab92005ca670fc536c99.jpg  
      inflating: ./clothes_dataset/white_pants/bd2b5b9605438bb1de2b293d31e41fb3942b4438.jpg  
      inflating: ./clothes_dataset/white_pants/c17733fd30a202a336dc91ccedee3f8bedb94230.jpg  
      inflating: ./clothes_dataset/white_pants/c1fc6d06d5158eb2d0a272239f0d1a792e93782a.jpg  
      inflating: ./clothes_dataset/white_pants/c3d04c0759f1ef36a4eded039230d0f2b963840a.jpg  
      inflating: ./clothes_dataset/white_pants/c5723d91cc6551ccee62ed6600255f862d5afe04.jpg  
      inflating: ./clothes_dataset/white_pants/c5e702ae3bcd631e9630546f36ddca1b3d26e465.jpg  
      inflating: ./clothes_dataset/white_pants/c68d2b6d9ce6772a3266d12f51499d576ef546de.jpg  
      inflating: ./clothes_dataset/white_pants/c7f56fe7621a6c1fe9ef93a2b0fd7df44af9411e.jpg  
      inflating: ./clothes_dataset/white_pants/c82ddc0e804d1332ec9ed5f7dc3af079500fc645.jpg  
      inflating: ./clothes_dataset/white_pants/c88aefb758ea6a921c61ecdbb37bd7f2d505b7e2.jpg  
      inflating: ./clothes_dataset/white_pants/c8fee3bd50983e3c361ad31a65509437ebe71133.jpg  
      inflating: ./clothes_dataset/white_pants/c94e85473622a9e5ba496c5fab5ab7d44e5aa807.jpg  
      inflating: ./clothes_dataset/white_pants/c966c2def0ff641b70fe40374553a8f41196f773.jpg  
      inflating: ./clothes_dataset/white_pants/c972de8d42cacf010a9a6429772bc71718e3fc9e.jpg  
      inflating: ./clothes_dataset/white_pants/c9eb8cd4dcbe1eecb82b87dce9d04d714af64f3d.jpg  
      inflating: ./clothes_dataset/white_pants/caf3ba40e8b0e9001e2094356da87b457f4ed714.jpg  
      inflating: ./clothes_dataset/white_pants/cc6c1af578ebf49ecfd6bfe9cff52b4b3f9e60b9.jpg  
      inflating: ./clothes_dataset/white_pants/cd61fe97009e00279be9576e4c69941b404a2445.jpg  
      inflating: ./clothes_dataset/white_pants/ce19fc65c002626a5962c6663041c915ad40b2f6.jpg  
      inflating: ./clothes_dataset/white_pants/cf198276907fc1b0ce4e2654be77e86a5d64b65c.jpg  
      inflating: ./clothes_dataset/white_pants/d2014bf506c4306f8af84b8a1a1912886dcc4c1b.jpg  
      inflating: ./clothes_dataset/white_pants/d29cd50ae80ba7c4fe3a4c1b52e3216ef2a689ae.jpg  
      inflating: ./clothes_dataset/white_pants/d33cd37a29ead87ce879d64d541a3400549d39f6.jpg  
      inflating: ./clothes_dataset/white_pants/d364b336fc6a02c07c46643f40f8523ad9a6b086.jpg  
      inflating: ./clothes_dataset/white_pants/d614720d6e7a94c896a27f0e7c73c7d242a5a9cf.jpg  
      inflating: ./clothes_dataset/white_pants/d6a90e8a38856272b4d9fdaf888ac35c98ccee7d.jpg  
      inflating: ./clothes_dataset/white_pants/d6b6653afe7069ad37c66ab3e48816db4fcff702.jpg  
      inflating: ./clothes_dataset/white_pants/d7216f93ed897ad48594d43ab0db8222865c1413.jpg  
      inflating: ./clothes_dataset/white_pants/d7d7c9d397fb8c07f5663ec54e3c74dcf35682d8.jpg  
      inflating: ./clothes_dataset/white_pants/d893da87a297bb52400b0c7a8eca94341d306a15.jpg  
      inflating: ./clothes_dataset/white_pants/d9b9beacc0866bbac8e27d9c9212e6ef093b237c.jpg  
      inflating: ./clothes_dataset/white_pants/da4cd3d34c2b8631266f3fbdda825255c1ddbc74.jpg  
      inflating: ./clothes_dataset/white_pants/daa78214989a7f25ba867bd34bea59b0c802fe63.jpg  
      inflating: ./clothes_dataset/white_pants/dafab34a6f81f11e17767f428da448a4f03550a2.jpg  
      inflating: ./clothes_dataset/white_pants/dbc26f415dae2c77c7da3d29cbefae6f12956d26.jpg  
      inflating: ./clothes_dataset/white_pants/ddfee6dfb16fb9b7744321e424619fca985ce89f.jpg  
      inflating: ./clothes_dataset/white_pants/de9110c61e9bf02e06dce5f2cb675a48b6eff075.jpg  
      inflating: ./clothes_dataset/white_pants/dec04668cfe165028c49bcbe309d018ff2a07d1b.jpg  
      inflating: ./clothes_dataset/white_pants/df174a6726f6e729d039c3a56275ddf6485538b6.jpg  
      inflating: ./clothes_dataset/white_pants/df52e39dda2ae530982a3cf455e23885964c12a2.jpg  
      inflating: ./clothes_dataset/white_pants/dfd5fe13b92e4cb8618a7e0f27d26d6262c5117a.jpg  
      inflating: ./clothes_dataset/white_pants/e147df857ad7d04b204b928e4e68108484c07460.jpg  
      inflating: ./clothes_dataset/white_pants/e21ba4f711c23c449327b66a4b454c4b1e441fde.jpg  
      inflating: ./clothes_dataset/white_pants/e400f4066b8d5ba4d13506513d0b6c0b897e91a3.jpg  
      inflating: ./clothes_dataset/white_pants/e68da964e217e77105901514b5cb6cf085bbfd02.jpg  
      inflating: ./clothes_dataset/white_pants/e97eeff470c2f027a1910d1322de8e1983d1bf8f.jpg  
      inflating: ./clothes_dataset/white_pants/ea0230fd1214f2cb7db637803610452d0d43525e.jpg  
      inflating: ./clothes_dataset/white_pants/ea6a170e127a0427ed09a4808200295aaf301b98.jpg  
      inflating: ./clothes_dataset/white_pants/ee55d5c53622b45628a3824c24e265c5c1aaa031.jpg  
      inflating: ./clothes_dataset/white_pants/efdb04cf808dd0f8e9406a92349dbfda5792351a.jpg  
      inflating: ./clothes_dataset/white_pants/f01e12e8709505d8828bf86151000327bc4b737c.jpg  
      inflating: ./clothes_dataset/white_pants/f09c1dd7ab3522e0a56f1dc01c40b21ece0f436e.jpg  
      inflating: ./clothes_dataset/white_pants/f18eccdadfce89222738a502689beef39fb0207d.jpg  
      inflating: ./clothes_dataset/white_pants/f4144a1a6ebe17118373664cf3886014b7ee08d8.jpg  
      inflating: ./clothes_dataset/white_pants/f527574dea6871376652a20044cd96f1749ee4f1.jpg  
      inflating: ./clothes_dataset/white_pants/f5885da50569abd77dd78419f8164d8bd8ba693d.jpg  
      inflating: ./clothes_dataset/white_pants/f5a9000c4156c2bb11090fd6d52af852f2270ba2.jpg  
      inflating: ./clothes_dataset/white_pants/f6a291c7b9b81253737a28094e6835d6f624a788.jpg  
      inflating: ./clothes_dataset/white_pants/f6af557d6fed52e5ae9f0b415469447ff8047dd5.jpg  
      inflating: ./clothes_dataset/white_pants/f6cd7c7da2d890d1bc488af4f245c2464393529a.jpg  
      inflating: ./clothes_dataset/white_pants/f728b0c312dc5596918880b3f081d78c996b6747.jpg  
      inflating: ./clothes_dataset/white_pants/f905f7ad133f0f3635c9303c92b798a6320dbccf.jpg  
      inflating: ./clothes_dataset/white_pants/f97327a07831adbb27b0545f5df21184341abd84.jpg  
      inflating: ./clothes_dataset/white_pants/f9a6572494b9af4e01f390e08a28555b6311aefd.jpg  
      inflating: ./clothes_dataset/white_pants/f9aad82ac3c8aec4bb62b0e1af01c370668431f6.jpg  
      inflating: ./clothes_dataset/white_pants/fa3f99d2852d89860c5b752aa5f2efa883c1fea2.jpg  
      inflating: ./clothes_dataset/white_pants/fa87a285f459ac4537fa5ac9f8c35f95d2843d5e.jpg  
      inflating: ./clothes_dataset/white_pants/fbff32ffdd244ae6fba4c87a6ed44e92010ed36b.jpg  
      inflating: ./clothes_dataset/white_pants/fc8a7c4046b6a8370b9a91c4faec3e5a9a37a76e.jpg  
      inflating: ./clothes_dataset/white_pants/fcab28198d20ff497ca3575923326c8c4e695eb5.jpg  
      inflating: ./clothes_dataset/white_pants/fd209c3d3c719940bf8cb4c7bd31a596ac183ac6.jpg  
      inflating: ./clothes_dataset/white_pants/fe922941fad4f078324c10b1df353e4fddc2e9dd.jpg  
      inflating: ./clothes_dataset/white_pants/fefbe4f5287c0572fec8c145bc296d1f3681ead6.jpg  
      inflating: ./clothes_dataset/white_pants/ff664356c36cfdbeac30e7635f93d5e3e5c22b7e.jpg  
      inflating: ./clothes_dataset/white_shoes/00271191db9b1fd604064a480239651950d348de.jpg  
      inflating: ./clothes_dataset/white_shoes/020f3b42539b1efa08ca77d0c7aaca5a3c464778.jpg  
      inflating: ./clothes_dataset/white_shoes/023fa7dee7b5262d3cb9eb1efc7c9a2d17466e79.jpg  
      inflating: ./clothes_dataset/white_shoes/02535e255e304e617ebd00b2b7172390cfef005b.jpg  
      inflating: ./clothes_dataset/white_shoes/025c8464a359acfe1cafd036bfa8284ade632fff.jpg  
      inflating: ./clothes_dataset/white_shoes/034e6c467ae63f32610b319b6616a093c318c856.jpg  
      inflating: ./clothes_dataset/white_shoes/03a2b0fc733d4b4ae13ad5e7e3b2787c0163434c.jpg  
      inflating: ./clothes_dataset/white_shoes/03f1437760dc728c44c5bd9137b1b17a93efd61f.jpg  
      inflating: ./clothes_dataset/white_shoes/040cd1a69a899c4f945ee19ab1aae16462a22cf5.jpg  
      inflating: ./clothes_dataset/white_shoes/040ff717035e24e938d70b46e9fbb588956ee590.jpg  
      inflating: ./clothes_dataset/white_shoes/043a7248c70e0fac86d0bf9de97fb474d25e7672.jpg  
      inflating: ./clothes_dataset/white_shoes/047cf1c2d861c482b1c181f99c568bcc574d7041.jpg  
      inflating: ./clothes_dataset/white_shoes/0547b30be2464068bef630952fc05c132730cde8.jpg  
      inflating: ./clothes_dataset/white_shoes/05cca54f551e2583d3097fc4d396830fd68dc8e8.jpg  
      inflating: ./clothes_dataset/white_shoes/06be9309ea63a4a6e19da8281b08875374f35022.jpg  
      inflating: ./clothes_dataset/white_shoes/07361f88f6ed82108217b8812d9a7ba9fe2e3a2b.jpg  
      inflating: ./clothes_dataset/white_shoes/074d5c1b2c3054e28c2a70902e5eb3e5d42e2bab.jpg  
      inflating: ./clothes_dataset/white_shoes/07cc3db5d7794a7282696a3ffbf9e5123ed1ad83.jpg  
      inflating: ./clothes_dataset/white_shoes/07d1de3835939753c127621e3c0108943ffeee22.jpg  
      inflating: ./clothes_dataset/white_shoes/07dc25ff60c2a4281c7c1e9f10686b404df8cac5.jpg  
      inflating: ./clothes_dataset/white_shoes/0870716d8fb886ca5ddcbc2080605e4f0a29b465.jpg  
      inflating: ./clothes_dataset/white_shoes/08853f49933acd74a38909d9ebde531e6ff31275.jpg  
      inflating: ./clothes_dataset/white_shoes/08db02cd8cbf79db86aff4a49c6d933e641d0c75.jpg  
      inflating: ./clothes_dataset/white_shoes/09b0b582072008d1586637645c240f9a476c1b79.jpg  
      inflating: ./clothes_dataset/white_shoes/0bae63b58f21b725fdd87d48d15a045c5896ff9c.jpg  
      inflating: ./clothes_dataset/white_shoes/0bcbbde9c416b0c2f5b427e681b78adcf4924f4a.jpg  
      inflating: ./clothes_dataset/white_shoes/0bd5d9f6466d99bdcf3e2b99fd6cfa1a3ed1b08b.jpg  
      inflating: ./clothes_dataset/white_shoes/0caba287be658380813ece8607bfc4b44fa895e4.jpg  
      inflating: ./clothes_dataset/white_shoes/0d6041e45cd6c21a9094eec5ddec806b044c78fe.jpg  
      inflating: ./clothes_dataset/white_shoes/0d631e7619db054047d42a197980659bc99601d4.jpg  
      inflating: ./clothes_dataset/white_shoes/0d6d4d0047f3c83f369f7c0ec91848b909083dee.jpg  
      inflating: ./clothes_dataset/white_shoes/0dbca69ae66267621b80bf5b5cb88d55e1926d73.jpg  
      inflating: ./clothes_dataset/white_shoes/0e0f8dbbe778063bbde756446c60b9bedf16ef35.jpg  
      inflating: ./clothes_dataset/white_shoes/0ef8be9a48bf98054571cd7aaa61865619f92b11.jpg  
      inflating: ./clothes_dataset/white_shoes/0f4b51e2c7877a4b9b9e44d82969bd38febd4736.jpg  
      inflating: ./clothes_dataset/white_shoes/0f6063097bd9629af00d330a757ea3ff03015612.jpg  
      inflating: ./clothes_dataset/white_shoes/0f6c192fbffc130daf13fe11b73f5e89740bf5fc.jpg  
      inflating: ./clothes_dataset/white_shoes/0f804530d83356699249c82209b4ab43816d440b.jpg  
      inflating: ./clothes_dataset/white_shoes/0f8f9430e864c381efbba0a2a729089f812daa17.jpg  
      inflating: ./clothes_dataset/white_shoes/102de904815a391a5293ce8593223d1e877ce708.jpg  
      inflating: ./clothes_dataset/white_shoes/106c9d3f875e517c80c2329158d218ee134ef5f3.jpg  
      inflating: ./clothes_dataset/white_shoes/1096a8234570f35714f8d8be63b94f83ae6c4101.jpg  
      inflating: ./clothes_dataset/white_shoes/11281130b07edfa26a2a7ec5c7d58950f9d89699.jpg  
      inflating: ./clothes_dataset/white_shoes/113aa579e5a4758c310c6322fede884764db7fc8.jpg  
      inflating: ./clothes_dataset/white_shoes/119aa9844883c26d413b32f464240d81b62e22e4.jpg  
      inflating: ./clothes_dataset/white_shoes/11c4662f56d1eeac511df96f915a424b3697b488.jpg  
      inflating: ./clothes_dataset/white_shoes/11ea2a1a58ac4406e49119bab15337262ceaa11a.jpg  
      inflating: ./clothes_dataset/white_shoes/12262ac1614bc5a6a232d79be02ca36a3129311e.jpg  
      inflating: ./clothes_dataset/white_shoes/1238146de792d9a2a35fac50da008f082e4c1fe7.jpg  
      inflating: ./clothes_dataset/white_shoes/1294986866636fe2dc25105f3a1e87982b654d32.jpg  
      inflating: ./clothes_dataset/white_shoes/12eba09ffcc4d7b9bef770221d0e008e5043d79c.jpg  
      inflating: ./clothes_dataset/white_shoes/140331f112cf8aa4c2e2e8f245e16ed74825c5d8.jpg  
      inflating: ./clothes_dataset/white_shoes/1465f00a146452a54a13f476ac32c377e0cedc56.jpg  
      inflating: ./clothes_dataset/white_shoes/14a31f19852ca81541b41245edf2634e3a59873c.jpg  
      inflating: ./clothes_dataset/white_shoes/14d60614de7d37b6944882462d25058af8656728.jpg  
      inflating: ./clothes_dataset/white_shoes/14f497e38464d75bbbd8de853351636dd818c8df.jpg  
      inflating: ./clothes_dataset/white_shoes/14f959c873633f0936fa0f14d4471924a2cd3fd8.jpg  
      inflating: ./clothes_dataset/white_shoes/154bc5e48838e287022e138af0b071ac9d83ba2c.jpg  
      inflating: ./clothes_dataset/white_shoes/1583fb3e6b53fe99dd50a09f382ab0f276cc83c5.jpg  
      inflating: ./clothes_dataset/white_shoes/1596da081fb7499b497943cb15ba9f0348d5fbba.jpg  
      inflating: ./clothes_dataset/white_shoes/15c01f0532e8fc6540358cf53af9694f190ff575.jpg  
      inflating: ./clothes_dataset/white_shoes/15c769f3566823d805c2ff5b1769ae328472ac4d.jpg  
      inflating: ./clothes_dataset/white_shoes/163fcdc724f339d3ac63f7117d14787463059bec.jpg  
      inflating: ./clothes_dataset/white_shoes/1752806d22e8961fb8a083ea2102be812375da0a.jpg  
      inflating: ./clothes_dataset/white_shoes/17c9b55475a893898fad7fdb9608a4814809ddb6.jpg  
      inflating: ./clothes_dataset/white_shoes/1816d313a47b46e558a98189a5b0af6bd960835c.jpg  
      inflating: ./clothes_dataset/white_shoes/184411a14ad282690d455d47ce2bbc0cffdeec21.jpg  
      inflating: ./clothes_dataset/white_shoes/189e1ec351ef8d51bd82eac4095fbc09970cbfc5.jpg  
      inflating: ./clothes_dataset/white_shoes/1900bd9e009cdbd6a0d86619476fd759bd8046dc.jpg  
      inflating: ./clothes_dataset/white_shoes/19751b8046fe0479413a6635585fc88ce968b79b.jpg  
      inflating: ./clothes_dataset/white_shoes/19b92218f4d1bb276020a361346191512a077eb4.jpg  
      inflating: ./clothes_dataset/white_shoes/1a428b962514c97b9be8b715998550a90a2aed46.jpg  
      inflating: ./clothes_dataset/white_shoes/1a521dd8d5778705b12d39e7bb1c290c8ff239f7.jpg  
      inflating: ./clothes_dataset/white_shoes/1c2c9fc1c785dfa41c80c3a97b7745176bd8112c.jpg  
      inflating: ./clothes_dataset/white_shoes/1c3c5d8067c3f59f44f4253edf666284930b86a4.jpg  
      inflating: ./clothes_dataset/white_shoes/1c76fb89600c83729a9a55ec46be3e9733a0a2f1.jpg  
      inflating: ./clothes_dataset/white_shoes/1ccabf41dbde05f380735a1fe899a719df1952b5.jpg  
      inflating: ./clothes_dataset/white_shoes/1ce54f10387936b5a1e76860b477812180d457a8.jpg  
      inflating: ./clothes_dataset/white_shoes/1d6b5e504c81c13f7fbce17c5809cfe64fda9434.jpg  
      inflating: ./clothes_dataset/white_shoes/1d70ebaec12e09f3561bf77b3bbf30c55c0a6f53.jpg  
      inflating: ./clothes_dataset/white_shoes/1d872f76bef43c4703b6f72478e29fdc11bb7da5.jpg  
      inflating: ./clothes_dataset/white_shoes/1dc10ced8cfed9c23f807fa57b4b80664000a4fc.jpg  
      inflating: ./clothes_dataset/white_shoes/1ed10a5f029b15680f16b60544b0c8671c7c728c.jpg  
      inflating: ./clothes_dataset/white_shoes/1edb92016ec4937d29988704a0299239132d19ed.jpg  
      inflating: ./clothes_dataset/white_shoes/20773d8332e75dc6baa1f1486d22f5d99119d963.jpg  
      inflating: ./clothes_dataset/white_shoes/20c5dd2326da862cfd7c11f603855e0c0a966c21.jpg  
      inflating: ./clothes_dataset/white_shoes/21371ecabc48a189d423401edb7729cb8c29861e.jpg  
      inflating: ./clothes_dataset/white_shoes/22182f5277550dcd51536897a94c6a56371176e1.jpg  
      inflating: ./clothes_dataset/white_shoes/22a9807e01fecf6b95564a41e2a721c7298f611b.jpg  
      inflating: ./clothes_dataset/white_shoes/22c31ba9600d87bad0431bcff1ab858dd84be40c.jpg  
      inflating: ./clothes_dataset/white_shoes/22e08eeb8b5d837fd3cb82345aa3b32806bcf35c.jpg  
      inflating: ./clothes_dataset/white_shoes/232fb39dfe76177d8a4e397f889c6aa8cf16e962.jpg  
      inflating: ./clothes_dataset/white_shoes/23674f014dd96fb056eb2ae9868afd210bf01808.jpg  
      inflating: ./clothes_dataset/white_shoes/240489b9d3c96832ba48c86b7170f0c0836207ea.jpg  
      inflating: ./clothes_dataset/white_shoes/24b39793052a6bd37701e374447b37192de9364c.jpg  
      inflating: ./clothes_dataset/white_shoes/271ca050c04400741a4bfd012d68b7902c07969b.jpg  
      inflating: ./clothes_dataset/white_shoes/275b4c3d22f8a547186fafeb3d3928bccb208c22.jpg  
      inflating: ./clothes_dataset/white_shoes/2780972d202e6813aa13efa8963c626d07015744.jpg  
      inflating: ./clothes_dataset/white_shoes/2788857556ab943d3ca230ecf900d76724706f31.jpg  
      inflating: ./clothes_dataset/white_shoes/27b631f4fa529689e9da569446119656ecd85233.jpg  
      inflating: ./clothes_dataset/white_shoes/2815b67507ede8db50667dd04e425622b4752aef.jpg  
      inflating: ./clothes_dataset/white_shoes/2838724c965b7bc48e31dfcec2db6fa7c00c96c7.jpg  
      inflating: ./clothes_dataset/white_shoes/2995832bfd96288c5c685475e8bcba76ffdb930a.jpg  
      inflating: ./clothes_dataset/white_shoes/2a84ad9fe90e18be9dbebd7fa76b484646082cb5.jpg  
      inflating: ./clothes_dataset/white_shoes/2b3efd3aa5ff77dbb65224958895ae16313e733b.jpg  
      inflating: ./clothes_dataset/white_shoes/2cb124041212a510fa6554add21a04fa0094d7ab.jpg  
      inflating: ./clothes_dataset/white_shoes/2cb41b91765070ad29a430206455baa525ecada3.jpg  
      inflating: ./clothes_dataset/white_shoes/2d6dd6e1dd4fc97c24eb850c3f6385cf638630e4.jpg  
      inflating: ./clothes_dataset/white_shoes/2dc21806c57062bc3e23542961cf42ce7bfee192.jpg  
      inflating: ./clothes_dataset/white_shoes/2e2d22e5c98dd77b8a253c9d055b0242ac6b0c39.jpg  
      inflating: ./clothes_dataset/white_shoes/2eeb1e10b78da6d39436eb300ba15c825c77ce26.jpg  
      inflating: ./clothes_dataset/white_shoes/2ef2c10dad4a41f9d207b9cb829af954b7655ff9.jpg  
      inflating: ./clothes_dataset/white_shoes/2f5b70daaf47871f0a55a14ecf0c163c1c7a9e17.jpg  
      inflating: ./clothes_dataset/white_shoes/2fdcd87a8d49726614ed1778b4d5960bfe8ba13a.jpg  
      inflating: ./clothes_dataset/white_shoes/310cbbfc43ce231ac936a7a2b81897f932175250.jpg  
      inflating: ./clothes_dataset/white_shoes/323325864d593952261416ecd24b40b8a7ad54ea.jpg  
      inflating: ./clothes_dataset/white_shoes/32c8ef537815b138707d1dfadb57c24af704ff73.jpg  
      inflating: ./clothes_dataset/white_shoes/346516816dab086a78c0dca46781b2403838ff59.jpg  
      inflating: ./clothes_dataset/white_shoes/3501dd7b939aca6315b82dedfe5d4e3359f790f2.jpg  
      inflating: ./clothes_dataset/white_shoes/350f0d9fc00b5df4a5a09f399cc22d2cf9a36f78.jpg  
      inflating: ./clothes_dataset/white_shoes/35c2522f51f29a4a396c623391da5ee30b8c4a28.jpg  
      inflating: ./clothes_dataset/white_shoes/35d40368cb49964e9d36b2a28f7618fd15232567.jpg  
      inflating: ./clothes_dataset/white_shoes/36161ad82c2f314be3c2b868628022d18d8b7fee.jpg  
      inflating: ./clothes_dataset/white_shoes/36e474163159ebb645bf3afc204232050551babd.jpg  
      inflating: ./clothes_dataset/white_shoes/37734ad9467aaeb275cc85e779d80eb9dd4e4cb9.jpg  
      inflating: ./clothes_dataset/white_shoes/37b450947ddfd44589851fc7999916743e3b6873.jpg  
      inflating: ./clothes_dataset/white_shoes/385d104bab4388ec5d444001be7a1c29268623a9.jpg  
      inflating: ./clothes_dataset/white_shoes/3864040fc17623a2409ec3a88f1a0d9103a23c15.jpg  
      inflating: ./clothes_dataset/white_shoes/38b088864957fd576e6a1ecebbbb9402cfe5c510.jpg  
      inflating: ./clothes_dataset/white_shoes/3957eb7e9736a87ec8b910760bc7b11851c16a9d.jpg  
      inflating: ./clothes_dataset/white_shoes/39c02e1abede261636c62e6fd3d4b2c2769c2feb.jpg  
      inflating: ./clothes_dataset/white_shoes/39d7c989d157dc06ef0e26066583398e33ae2819.jpg  
      inflating: ./clothes_dataset/white_shoes/3a614975ca5114c222d3801a2c766b0a84265ae3.jpg  
      inflating: ./clothes_dataset/white_shoes/3b8c0e95885df39355c2ab69a28ac1a69b851808.jpg  
      inflating: ./clothes_dataset/white_shoes/3c4827b6ed82b0b331430fff03048bcf7c0c3a67.jpg  
      inflating: ./clothes_dataset/white_shoes/3c6afec4b8893bf696b822298bd592fcedbda93f.jpg  
      inflating: ./clothes_dataset/white_shoes/3d15646615959c2776958788ed589051710c5aae.jpg  
      inflating: ./clothes_dataset/white_shoes/3f1e3098848eb3c252f79efd009ca81db4f4ac97.jpg  
      inflating: ./clothes_dataset/white_shoes/3f5dad437e4136e8a6031bec4795f3a2fb3e535c.jpg  
      inflating: ./clothes_dataset/white_shoes/3fdc662e833c1879a2d1cbfaaa317e32f3b71dd3.jpg  
      inflating: ./clothes_dataset/white_shoes/40713ac43d9fd58df6fe56f37e7e3ac846936d1a.jpg  
      inflating: ./clothes_dataset/white_shoes/409d3e12825867fd9db03806ca4a493e0affa1af.jpg  
      inflating: ./clothes_dataset/white_shoes/40e9d65beacf079e9dae1e0436adb38ce5aab142.jpg  
      inflating: ./clothes_dataset/white_shoes/41158da8dabea56e66a4b2451bd468ae3e3cb50d.jpg  
      inflating: ./clothes_dataset/white_shoes/411d538f2d1d4aff188cc40e31fbeaa031eaa5b8.jpg  
      inflating: ./clothes_dataset/white_shoes/411d757eb70ce5219a90a51d0d83f448166c81de.jpg  
      inflating: ./clothes_dataset/white_shoes/42b1277b5e7b78535efdf337cf644de7da2b7414.jpg  
      inflating: ./clothes_dataset/white_shoes/42e4fcb4f286463a988653cc8718cf58b317ff1b.jpg  
      inflating: ./clothes_dataset/white_shoes/4360b67c19b0d5908a0432091e847bda36c05e83.jpg  
      inflating: ./clothes_dataset/white_shoes/444188c600753003876eff28003ee1ec4bb5805d.jpg  
      inflating: ./clothes_dataset/white_shoes/4471c94d179501b2dff8a0e13e9d263361ad4037.jpg  
      inflating: ./clothes_dataset/white_shoes/4545ce7dcf2779a7c7cae5c5cba7ec9de18619d0.jpg  
      inflating: ./clothes_dataset/white_shoes/4560551a1df1ddf1a16f12e275e24f72e35567eb.jpg  
      inflating: ./clothes_dataset/white_shoes/45a95c5227c207fe657d9959cc6e3f89d5d33559.jpg  
      inflating: ./clothes_dataset/white_shoes/45bbb984b98eb85abd618a824a0efe8eded78b64.jpg  
      inflating: ./clothes_dataset/white_shoes/45e49cadaf056eea7175b43c1f6eb1a561fa53eb.jpg  
      inflating: ./clothes_dataset/white_shoes/45ee56656da901096ccff18c869cad090fa2cc44.jpg  
      inflating: ./clothes_dataset/white_shoes/45f2f6dce7f55d733dc0958eb88b058222dc9c04.jpg  
      inflating: ./clothes_dataset/white_shoes/47a354840bc9e9b63ac958813135af860d9e26f3.jpg  
      inflating: ./clothes_dataset/white_shoes/47dbba84d921fc274b6f1b3ce8a8c3099cd248ab.jpg  
      inflating: ./clothes_dataset/white_shoes/4850181b45f83fff23cd31e0f5fd006b8f44b6ae.jpg  
      inflating: ./clothes_dataset/white_shoes/4896e2e1d9a155041337976f7855402c427dc6ed.jpg  
      inflating: ./clothes_dataset/white_shoes/494e9d28ea5cd9d3ea839c47b4016b81655aa8e8.jpg  
      inflating: ./clothes_dataset/white_shoes/4997a2c97f2d6cf3ea1af082c94dbbd3a4ba5adf.jpg  
      inflating: ./clothes_dataset/white_shoes/49bbad327b6a372763a83f92c8bbc6b8a35ab9ea.jpg  
      inflating: ./clothes_dataset/white_shoes/49ce9ddaa65c3909b6098065cfcbb22b4923c912.jpg  
      inflating: ./clothes_dataset/white_shoes/4a5a16048f0842e2e2df0f67ace66bb14a5f32b9.jpg  
      inflating: ./clothes_dataset/white_shoes/4a8b871914f36e0e1d22595f99e332f598496f2e.jpg  
      inflating: ./clothes_dataset/white_shoes/4b4f6295d4024826e595d2d257607d3adf42c19e.jpg  
      inflating: ./clothes_dataset/white_shoes/4b5c7916c79c813377dacc7d651fd9e6578d2e27.jpg  
      inflating: ./clothes_dataset/white_shoes/4b74c39bdbce059d2c44482471083a853373f806.jpg  
      inflating: ./clothes_dataset/white_shoes/4b7b400c02e861d82ab07feda6f67812ec4f2c62.jpg  
      inflating: ./clothes_dataset/white_shoes/4ba6a2791a3e5419b1201680989ae882e129ae00.jpg  
      inflating: ./clothes_dataset/white_shoes/4bbf3080fc2c827a742424bb5f2853bc9e9fcad7.jpg  
      inflating: ./clothes_dataset/white_shoes/4c0499124829ed258f1796d31bac4405e9dbc160.jpg  
      inflating: ./clothes_dataset/white_shoes/4d99e99845d2b969633ba7d8536cbe353bf454e5.jpg  
      inflating: ./clothes_dataset/white_shoes/4dbff3c9d7c937444525d01b0ca9ad18be4a8155.jpg  
      inflating: ./clothes_dataset/white_shoes/4e16971db4daa2ad6205cf93cf92de143cd7abe8.jpg  
      inflating: ./clothes_dataset/white_shoes/4eaacd4947c8dc1536fcfa11f727e9e7d5a701e0.jpg  
      inflating: ./clothes_dataset/white_shoes/4f213f72d9853f24853a54cc462cce24687c89a0.jpg  
      inflating: ./clothes_dataset/white_shoes/4f61b74517140995fdf99e210ba93850faeb7cf9.jpg  
      inflating: ./clothes_dataset/white_shoes/4fa0b90f5c421f4aa8fdd742b751a62764a514fa.jpg  
      inflating: ./clothes_dataset/white_shoes/4fd35845fde5be3094b7634aeaebfba1f0e5736a.jpg  
      inflating: ./clothes_dataset/white_shoes/502391c8b2663f8e1fb15c5b56c04b52cd9bbc02.jpg  
      inflating: ./clothes_dataset/white_shoes/511907e42968b66075460119e7e6f429dc2eb92c.jpg  
      inflating: ./clothes_dataset/white_shoes/5125b8e6417c6f6c43bc47c9e6004d775ee48922.jpg  
      inflating: ./clothes_dataset/white_shoes/5126f539bd766c7a08548beb541464cc44995eaa.jpg  
      inflating: ./clothes_dataset/white_shoes/5187f4988cd59a424642de2e1bf40f17555da6a1.jpg  
      inflating: ./clothes_dataset/white_shoes/523d59f493886537cda349e2c22e4ee86633b2cf.jpg  
      inflating: ./clothes_dataset/white_shoes/525f1cdf986d383ff0608ad6632705d48e71356f.jpg  
      inflating: ./clothes_dataset/white_shoes/5307f094623ad2f566ffa59d67b8a7be3bcaf773.jpg  
      inflating: ./clothes_dataset/white_shoes/535f8df3bf321fbcec3fbb876bc9444c106a4653.jpg  
      inflating: ./clothes_dataset/white_shoes/537fc15949b37034d62bd697570c09d0c1917360.jpg  
      inflating: ./clothes_dataset/white_shoes/53be52ebc30af9f1929106dc84cb637febc92eb4.jpg  
      inflating: ./clothes_dataset/white_shoes/53cd81658ba6dfaf61564222072161c0ca5dc6ba.jpg  
      inflating: ./clothes_dataset/white_shoes/53d7a702facde8c2d6fbd5f26ee29a10dc8050fc.jpg  
      inflating: ./clothes_dataset/white_shoes/53e5750d25c374e2e12ff259b4410507528946f4.jpg  
      inflating: ./clothes_dataset/white_shoes/54854a862430bee389e028036bfd662a5b4ebef1.jpg  
      inflating: ./clothes_dataset/white_shoes/5545e1b56d9316fc8db5194c8b2600c43b5b36c5.jpg  
      inflating: ./clothes_dataset/white_shoes/5617188b51142c5636c113dbce0e830da5b839fd.jpg  
      inflating: ./clothes_dataset/white_shoes/56ae8f049851be825a74a890a7b0c1aa0a4f3f9d.jpg  
      inflating: ./clothes_dataset/white_shoes/56c818ca32f2a21cc01de24cff9d8c0fe7735429.jpg  
      inflating: ./clothes_dataset/white_shoes/56de0e52ae2aa962344933624259ee4672eddcb2.jpg  
      inflating: ./clothes_dataset/white_shoes/56eae3410436295355046f8205b37f8730f3c334.jpg  
      inflating: ./clothes_dataset/white_shoes/56ed25a4daed13cd1df2bfb6e2b9dbeaf546aeb5.jpg  
      inflating: ./clothes_dataset/white_shoes/5738ff235d00afb22a73950de8c01baa603ee4c6.jpg  
      inflating: ./clothes_dataset/white_shoes/57adbea03610edea7cb55188bc28449083c71bfb.jpg  
      inflating: ./clothes_dataset/white_shoes/581e5775cfb6327f164ba340d4d310e4a75e9169.jpg  
      inflating: ./clothes_dataset/white_shoes/58b838c93e2cd261b4158639f57de7442656babc.jpg  
      inflating: ./clothes_dataset/white_shoes/58db91dff8856e45014168ec99ca2303d8e99540.jpg  
      inflating: ./clothes_dataset/white_shoes/591d8befc239c345fc6105b38b3e368968424401.jpg  
      inflating: ./clothes_dataset/white_shoes/59803fb01d162db958d15c2707c71ed11028648f.jpg  
      inflating: ./clothes_dataset/white_shoes/599d33c2b7b1cabbe13331a081f707988dfe0035.jpg  
      inflating: ./clothes_dataset/white_shoes/59b5a635c9bd0768e529205b4ced0ce6ca7eed43.jpg  
      inflating: ./clothes_dataset/white_shoes/59df445905d1ccbd8627a54451d8752960ed31fd.jpg  
      inflating: ./clothes_dataset/white_shoes/5a039796e0ac100bf3ed2f73294e123d42e26c23.jpg  
      inflating: ./clothes_dataset/white_shoes/5a742bc5b2e8ba03cc98e16bd6004c6837ef7419.jpg  
      inflating: ./clothes_dataset/white_shoes/5b9a42939cadb2f76fc8aaf2f4ea9f51d1899e74.jpg  
      inflating: ./clothes_dataset/white_shoes/5beac72aeaf5b77cff6025d7297ce34103863a33.jpg  
      inflating: ./clothes_dataset/white_shoes/5c80befba80cdae63ccd9609980dc611e549df8a.jpg  
      inflating: ./clothes_dataset/white_shoes/5d1fea8fda024c9dd0b2c691cc5267bfe49b4400.jpg  
      inflating: ./clothes_dataset/white_shoes/5e01546021e46ccbba1d2f9f75ecb7685ec347c3.jpg  
      inflating: ./clothes_dataset/white_shoes/5e466b0dfe05c2160a09f787900c8ea1bee959f7.jpg  
      inflating: ./clothes_dataset/white_shoes/5e90e7e362d66dffea1e24d25c72eb2722e5399b.jpg  
      inflating: ./clothes_dataset/white_shoes/5e917655d0a1459e9c477acfa5be62cf3e583c36.jpg  
      inflating: ./clothes_dataset/white_shoes/5e987553a913bf2e0c78d4bfd70643204bf8434e.jpg  
      inflating: ./clothes_dataset/white_shoes/5f33e2a73b879d1d1cd3a736eff0fd7e6ade4d15.jpg  
      inflating: ./clothes_dataset/white_shoes/5f5398c972d7f38bd2c2f751a2964308a934df76.jpg  
      inflating: ./clothes_dataset/white_shoes/5f6b1f455a82c8bc0de3008b77ed09873dfda4e2.jpg  
      inflating: ./clothes_dataset/white_shoes/5fac68e4a8d88c77dbdb39d5188a9c8671505e51.jpg  
      inflating: ./clothes_dataset/white_shoes/5fde1a7dc298b7126aa0c6b9d1f8e8471e12f3d6.jpg  
      inflating: ./clothes_dataset/white_shoes/601b48e9a4dbd4607d60527216a68b67866594e3.jpg  
      inflating: ./clothes_dataset/white_shoes/6062c4e454e35705478182c966685eb0738ff048.jpg  
      inflating: ./clothes_dataset/white_shoes/60e6753411a26419d7a3f7ae1bade890904ebf0c.jpg  
      inflating: ./clothes_dataset/white_shoes/60f35901fe4ae14f3f4a7c73a6b7f8d15aba1046.jpg  
      inflating: ./clothes_dataset/white_shoes/61019663113ef5f0fe71450ab5e28e72c9e25128.jpg  
      inflating: ./clothes_dataset/white_shoes/610e15c00019d3f63935c34e87f9d9decc767d2a.jpg  
      inflating: ./clothes_dataset/white_shoes/61175b68c87cb5086407a3e343d209149f865931.jpg  
      inflating: ./clothes_dataset/white_shoes/6154110a9dc7d9c54471e164c08d4f99cd5f8189.jpg  
      inflating: ./clothes_dataset/white_shoes/61b7850ccf02af0168d82768b1a4d2a948cfa377.jpg  
      inflating: ./clothes_dataset/white_shoes/6291896ec4e0005630c6d1c9f22d196978d64f62.jpg  
      inflating: ./clothes_dataset/white_shoes/6327259b3358e04ccdba22f477b0547d51a344f5.jpg  
      inflating: ./clothes_dataset/white_shoes/6366f4a80412cb6b43dba45ab3975548ae2bf397.jpg  
      inflating: ./clothes_dataset/white_shoes/63a97914137c39d765f9d3cf513d58bbfe0a3664.jpg  
      inflating: ./clothes_dataset/white_shoes/6433f4d933da5f03c9801178503867bbf5339bf7.jpg  
      inflating: ./clothes_dataset/white_shoes/65d732e4cc4e41b0599d5886aabcdab32c747d5d.jpg  
      inflating: ./clothes_dataset/white_shoes/66967835c384d80ce596c58af4352d1dce7c4614.jpg  
      inflating: ./clothes_dataset/white_shoes/66e01122ad2f51f3120243fd589bd880ed99e5df.jpg  
      inflating: ./clothes_dataset/white_shoes/685a1fe5f53b1c742d1ecea091569d1cdbddef2a.jpg  
      inflating: ./clothes_dataset/white_shoes/68e074894c5809193656cbea4fe2e747a0861b25.jpg  
      inflating: ./clothes_dataset/white_shoes/68e5a877b8a18e823776d55ecdbab7d7a45d09a0.jpg  
      inflating: ./clothes_dataset/white_shoes/68f8875ec338cddad80e191c78243991894a0c70.jpg  
      inflating: ./clothes_dataset/white_shoes/695c8183c429ad90273dd6c370d8161fd1012139.jpg  
      inflating: ./clothes_dataset/white_shoes/696e08fac4c047724bc55c150e4a36af29c7ff1f.jpg  
      inflating: ./clothes_dataset/white_shoes/6a03c5c3bd1760bda178ea352dc26b2b2b18851b.jpg  
      inflating: ./clothes_dataset/white_shoes/6aba869b043307769a67d4febd5a18c969fb0634.jpg  
      inflating: ./clothes_dataset/white_shoes/6b298d189b65159333fbe622ef5ae3e3b5e1711d.jpg  
      inflating: ./clothes_dataset/white_shoes/6b9502e2852a3ca7b07773ab96492ad0b80a1839.jpg  
      inflating: ./clothes_dataset/white_shoes/6bdb626847033b683c72e506db77eb177a54483c.jpg  
      inflating: ./clothes_dataset/white_shoes/6bf42c5497e72a9ab7ec933ba0a635b728f79f65.jpg  
      inflating: ./clothes_dataset/white_shoes/6c8eb05a7207a6f58c0c79ef7fe4b6ca2e7db493.jpg  
      inflating: ./clothes_dataset/white_shoes/6cd89f6f74e44c2853f619125896e7f315f32bae.jpg  
      inflating: ./clothes_dataset/white_shoes/6e13d213b09671c45c64dc58e574c6dce0bedc2d.jpg  
      inflating: ./clothes_dataset/white_shoes/6e25b9550995c5a9a49ec97af6d4b32f7261b193.jpg  
      inflating: ./clothes_dataset/white_shoes/6e71a0fc8f7cd0ec6b15a9cde1d8090ef5ddf201.jpg  
      inflating: ./clothes_dataset/white_shoes/6ed61bd06b9d319d6c89986f9adb683ed8ae06c0.jpg  
      inflating: ./clothes_dataset/white_shoes/6f460d240cb29a6cb91a72b5aecb33e976bbd1bb.jpg  
      inflating: ./clothes_dataset/white_shoes/6f86e04b08ef32e1e28447f8e52500d52b911e85.jpg  
      inflating: ./clothes_dataset/white_shoes/6fe9bdb776aeb80e69f9d4a316a10da3c461fbfe.jpg  
      inflating: ./clothes_dataset/white_shoes/7146e5f34c89ae6a657d4977c3bfa6a3bbdb96c3.jpg  
      inflating: ./clothes_dataset/white_shoes/723338cc9d98c3275ab6141af17293f02cee41c7.jpg  
      inflating: ./clothes_dataset/white_shoes/73a253e351f2023393914e11a56392d7e043b968.jpg  
      inflating: ./clothes_dataset/white_shoes/74840e796f3ac2b7097d39e6d816aa3d7faaf007.jpg  
      inflating: ./clothes_dataset/white_shoes/757b3b41fba3e01ca2a09b1c5662b7c49be6b892.jpg  
      inflating: ./clothes_dataset/white_shoes/759345a4ee111395774d6316ece850a58de0f6d4.jpg  
      inflating: ./clothes_dataset/white_shoes/7661d15ad1358be4a647e207bfe2b8c609e72b3c.jpg  
      inflating: ./clothes_dataset/white_shoes/768380f6d1dcb90629d80db5940843e9618ca260.jpg  
      inflating: ./clothes_dataset/white_shoes/76e15cbd5451242e1384003014e391fe8ecd3caf.jpg  
      inflating: ./clothes_dataset/white_shoes/77596c6fdd1c33ef4c5b098daa6edec4f8fa2ace.jpg  
      inflating: ./clothes_dataset/white_shoes/7864314e1ceb9598f866f878e6da1e8cee7e05f7.jpg  
      inflating: ./clothes_dataset/white_shoes/78c5066760aedf9aa1dde7cc2e419c9cf1643be5.jpg  
      inflating: ./clothes_dataset/white_shoes/7907403f8d40297fb98bc659037b2126750f203b.jpg  
      inflating: ./clothes_dataset/white_shoes/7981e5aac9be8637edfe7883b881c482be80ce97.jpg  
      inflating: ./clothes_dataset/white_shoes/7a32d5945171cc616bdb42c3f9bda7f16524dbe9.jpg  
      inflating: ./clothes_dataset/white_shoes/7b3f1304e5c437bc30361b2b00df2b88e26638f8.jpg  
      inflating: ./clothes_dataset/white_shoes/7ca8c4778b9170ba7835730f058ddf1407388ba1.jpg  
      inflating: ./clothes_dataset/white_shoes/7dce84d2d79e4e426d78119d5f3903e110cd44de.jpg  
      inflating: ./clothes_dataset/white_shoes/7e4041de2db8f47b6527d2ec3346289492706746.jpg  
      inflating: ./clothes_dataset/white_shoes/7e8446cc00a75711bfa26a0714da4e46b6c582f7.jpg  
      inflating: ./clothes_dataset/white_shoes/7e8d7c1bd6982da32f508a1deb88abb67bbd9122.jpg  
      inflating: ./clothes_dataset/white_shoes/7ed9e4148f91c95ca13a08134da6b7cc45706b78.jpg  
      inflating: ./clothes_dataset/white_shoes/7f0197132a74f07d2b8baa0d73ab2ac1584afd60.jpg  
      inflating: ./clothes_dataset/white_shoes/7f2a31fc70275af80009e0e029e45a8771ce91f8.jpg  
      inflating: ./clothes_dataset/white_shoes/7f8bc8c12e03860f677504a5602523bc9c8145e3.jpg  
      inflating: ./clothes_dataset/white_shoes/7f983ae64b382a105634a6c42e172357339eced3.jpg  
      inflating: ./clothes_dataset/white_shoes/803ea7f09a4af3777521d910b1dc2ea61614239f.jpg  
      inflating: ./clothes_dataset/white_shoes/810d8a4f191e3a60260a597107b0e36722d2512c.jpg  
      inflating: ./clothes_dataset/white_shoes/818f0936e0e70deb766eef35eb21710744f303ac.jpg  
      inflating: ./clothes_dataset/white_shoes/81b57aef6f30a744d9df81b355bfd08e012f221b.jpg  
      inflating: ./clothes_dataset/white_shoes/825b0dd05898cd3e1a804defbb786edd82819e1d.jpg  
      inflating: ./clothes_dataset/white_shoes/82a242947fce4c5901497601f7f764649cafd422.jpg  
      inflating: ./clothes_dataset/white_shoes/82eb1abf7afec2a25722e4b252ceaf41d1963e83.jpg  
      inflating: ./clothes_dataset/white_shoes/833963741d05474651f8332d8838ebcaffca39ba.jpg  
      inflating: ./clothes_dataset/white_shoes/838c2d9170ed3371688a2e4bb56a7397f4a86dbe.jpg  
      inflating: ./clothes_dataset/white_shoes/84a520911b27a8460f4bfd7fadbae709229a8909.jpg  
      inflating: ./clothes_dataset/white_shoes/85f0378fbfef2fe51be3410ad851274c153f243a.jpg  
      inflating: ./clothes_dataset/white_shoes/8660742ed13b1520d22ab4cc84e111c718c187a8.jpg  
      inflating: ./clothes_dataset/white_shoes/866b192dfdd30cd7ff46eae856cb4540b359f2b4.jpg  
      inflating: ./clothes_dataset/white_shoes/8717c8d2ce8c6e0e8fa77cc648ac0e443118786a.jpg  
      inflating: ./clothes_dataset/white_shoes/877e19b2691a0a312b1002a355de90431646c300.jpg  
      inflating: ./clothes_dataset/white_shoes/87d7be68e4d7cb1b5a3eca0cb5e9e16a8d179e1c.jpg  
      inflating: ./clothes_dataset/white_shoes/880e38f8f2bbbb75bb152cd989ef27b1023b48c3.jpg  
      inflating: ./clothes_dataset/white_shoes/883e6aeb73e4f269b6a95bf2ef1c6c7553aaa8d8.jpg  
      inflating: ./clothes_dataset/white_shoes/88a9948f89893cbd0d6275f3ef3673e82c8a3ce5.jpg  
      inflating: ./clothes_dataset/white_shoes/88d7201abaa086a1575e96c810eccad57d9d957c.jpg  
      inflating: ./clothes_dataset/white_shoes/89099c4402adc33699f0eaf3054e36cba866ae14.jpg  
      inflating: ./clothes_dataset/white_shoes/899b3f9c891273c6e82cecdbdc79179908737fbd.jpg  
      inflating: ./clothes_dataset/white_shoes/8a7cdfc31e9123957948b700e5ce1e8fc4af2641.jpg  
      inflating: ./clothes_dataset/white_shoes/8aff1d922be1371ddb3baeb02fd9c7b64b9b638f.jpg  
      inflating: ./clothes_dataset/white_shoes/8b7dbff3f3807bf126ccaceed15454b7d2f93293.jpg  
      inflating: ./clothes_dataset/white_shoes/8b92ebcfba796aee5d61254fbd317329fb9bd97e.jpg  
      inflating: ./clothes_dataset/white_shoes/8bee51ee7b6f36e4f30713c052e2a92c82ce2ff1.jpg  
      inflating: ./clothes_dataset/white_shoes/8c273700034b069367763cbae96141d550d31450.jpg  
      inflating: ./clothes_dataset/white_shoes/8c481d2745c3ef6bf48413f50316460774b4c9da.jpg  
      inflating: ./clothes_dataset/white_shoes/8c92e9a8a809bbd1b7fbd2d60e6fe21cdea78965.jpg  
      inflating: ./clothes_dataset/white_shoes/8cb94eeed13d740153f5e080fafa9caf0bb7c72d.jpg  
      inflating: ./clothes_dataset/white_shoes/8ce41f3cf92f6cb664370f771ea6265b301748ba.jpg  
      inflating: ./clothes_dataset/white_shoes/8db99757d28928ffd12709978d108ed0280ad336.jpg  
      inflating: ./clothes_dataset/white_shoes/8dee34143d8f25a9a4f3fbe8c7c455443bdf36ba.jpg  
      inflating: ./clothes_dataset/white_shoes/8e236497acdb0456f65a651ebc9ca183d7f769d8.jpg  
      inflating: ./clothes_dataset/white_shoes/8ec439adcf0a8fc9bbfb64d5ad0aaa721ac48d75.jpg  
      inflating: ./clothes_dataset/white_shoes/8f45aaa15760e81623771e36e8d26c53f6f31438.jpg  
      inflating: ./clothes_dataset/white_shoes/8f4aa0994d76d83bee3176d18cf50f66e9f7fa5a.jpg  
      inflating: ./clothes_dataset/white_shoes/8f5e908b00fda0a62d2116f8df75021df7b85cf2.jpg  
      inflating: ./clothes_dataset/white_shoes/8f8c3d3877b716e3726d74aa49d8eb5f6abf244a.jpg  
      inflating: ./clothes_dataset/white_shoes/8f949db54457415b361871f346a49071ecd6581d.jpg  
      inflating: ./clothes_dataset/white_shoes/9075d9a89e2c32029b256b778fdac67054cc5bc9.jpg  
      inflating: ./clothes_dataset/white_shoes/9170537a777a2a397002268ceff8bc83224d2297.jpg  
      inflating: ./clothes_dataset/white_shoes/9174687b61cb04509f62ef44a149f616a2345775.jpg  
      inflating: ./clothes_dataset/white_shoes/9195c2dc28ec2da17f25ba930c7ed43b763c7dbc.jpg  
      inflating: ./clothes_dataset/white_shoes/9205ec2159a08af9affbbaad2ca192654c1ea3b3.jpg  
      inflating: ./clothes_dataset/white_shoes/92232647a0ab4fbb2247984739333749a03918b9.jpg  
      inflating: ./clothes_dataset/white_shoes/940ff142e5ef2ebdbe1b3c8c6d3584126ae980ee.jpg  
      inflating: ./clothes_dataset/white_shoes/94b527873140ae9851b3169e9ddd784f4528f8d6.jpg  
      inflating: ./clothes_dataset/white_shoes/958f503b853cc70b7f07ecea72062df9abbc2814.jpg  
      inflating: ./clothes_dataset/white_shoes/962d2dc4ec4f9fe35da8d32465515cb9b46c7ab2.jpg  
      inflating: ./clothes_dataset/white_shoes/966a43b03d9c933af13eb3d3426e36efbfc81593.jpg  
      inflating: ./clothes_dataset/white_shoes/975cc6a5e25cf64ae63a592ad8075cddec87b0dc.jpg  
      inflating: ./clothes_dataset/white_shoes/975e269428fe194bde2217467d4f633fb4ac2a81.jpg  
      inflating: ./clothes_dataset/white_shoes/97815db632567f66afa29d7c441d476ab98777bf.jpg  
      inflating: ./clothes_dataset/white_shoes/99041406670dfc9f9ebacda2947acf7ae1d8a577.jpg  
      inflating: ./clothes_dataset/white_shoes/9963c12a5a7194af84393848da0dd300fca43cbd.jpg  
      inflating: ./clothes_dataset/white_shoes/99fa63496e6b8af24983ce34705d9b541013992d.jpg  
      inflating: ./clothes_dataset/white_shoes/9a6e31a96d06c06638e51c8259e1fb203df07253.jpg  
      inflating: ./clothes_dataset/white_shoes/9a8fc4332a5a0f15eeb02601da5292367d9d0a90.jpg  
      inflating: ./clothes_dataset/white_shoes/9b3e7cca11d8a9e91ff759f16e29bd64c20af25c.jpg  
      inflating: ./clothes_dataset/white_shoes/9b663fd53547d6c846e1c6ded98484cf3a74b945.jpg  
      inflating: ./clothes_dataset/white_shoes/9b821e065c50d952f00535de759c9b87a2b7c127.jpg  
      inflating: ./clothes_dataset/white_shoes/9c27cd9aaff873de335239e92c8a010c7109463c.jpg  
      inflating: ./clothes_dataset/white_shoes/9c59fde4ce1454e5c4a8ab64ed56013e4208ada3.jpg  
      inflating: ./clothes_dataset/white_shoes/9cea25c971e6bb06f9b56c124bbd4b3bd8b5d4d9.jpg  
      inflating: ./clothes_dataset/white_shoes/9cfe81e7ba8463add92f6b4883012eeebf05b3f9.jpg  
      inflating: ./clothes_dataset/white_shoes/9ddf09a920ded3bfb0fed6e452db64c01320069b.jpg  
      inflating: ./clothes_dataset/white_shoes/9e949c2eaa5ca7e36b8f658579548a92a82f5db5.jpg  
      inflating: ./clothes_dataset/white_shoes/9ed20252bcf1810b4dd6a2b382cee8840f603e5b.jpg  
      inflating: ./clothes_dataset/white_shoes/9fa440d8caae707e962f2323a017db22c9ebda71.jpg  
      inflating: ./clothes_dataset/white_shoes/a04ffdd809a6131d31d147444cdf513fa001b4c6.jpg  
      inflating: ./clothes_dataset/white_shoes/a05912ade4a3ade696a6e2e226f2fe046dee087c.jpg  
      inflating: ./clothes_dataset/white_shoes/a0a6f2bf5cec4da4aa78dda646b322d55a160a42.jpg  
      inflating: ./clothes_dataset/white_shoes/a0b8193dfb5c1b0d88b90aa35298438874f3b59d.jpg  
      inflating: ./clothes_dataset/white_shoes/a0bc6196c66c250ba216ba875b91a91c06b135e6.jpg  
      inflating: ./clothes_dataset/white_shoes/a0fbd1a2a11bd6b380a7463e95e08494551a9a2a.jpg  
      inflating: ./clothes_dataset/white_shoes/a1440d137cdf3c1d904aba97f5ae033383c87190.jpg  
      inflating: ./clothes_dataset/white_shoes/a1e4652b2b3100b4d3330f1275f437fd464e4337.jpg  
      inflating: ./clothes_dataset/white_shoes/a1e556834abf7fbbf0b1f4e9263d349434e85ca2.jpg  
      inflating: ./clothes_dataset/white_shoes/a2285c76c3774d07bb16e0152c744eb98fb1a562.jpg  
      inflating: ./clothes_dataset/white_shoes/a27563b11b4513185b25f8e5c298fb185535c09c.jpg  
      inflating: ./clothes_dataset/white_shoes/a29ec274a6e1de6ee9f58e97250f5579bce7c0ce.jpg  
      inflating: ./clothes_dataset/white_shoes/a2ba4348b0271fd480c0e41ce1c9341393c272ca.jpg  
      inflating: ./clothes_dataset/white_shoes/a3321407b5cd08151f1581bab9411561230c2421.jpg  
      inflating: ./clothes_dataset/white_shoes/a35ff51bdf31422398fd475fa06acb27947b9824.jpg  
      inflating: ./clothes_dataset/white_shoes/a38da80ceef860bb2e9e78fbe2a351df8fc2f150.jpg  
      inflating: ./clothes_dataset/white_shoes/a3ab14d6cf673006f1a9afd5ebaa80ae5138ac4c.jpg  
      inflating: ./clothes_dataset/white_shoes/a4332c18263401a89f4f91239c4303621c6b9d74.jpg  
      inflating: ./clothes_dataset/white_shoes/a460364b3c03f59499c12a2226676cff822ec0ff.jpg  
      inflating: ./clothes_dataset/white_shoes/a4822d54c54a1702c335088a49fa062da42db116.jpg  
      inflating: ./clothes_dataset/white_shoes/a4864a52d8c0a2543ea3554f3c71bdf74d9a2332.jpg  
      inflating: ./clothes_dataset/white_shoes/a4c947ce3cb985055de352307c913f7837a2a278.jpg  
      inflating: ./clothes_dataset/white_shoes/a61a10a7700d56a37afb730c5ee2cb4dea4fe929.jpg  
      inflating: ./clothes_dataset/white_shoes/a92e26c707976c45c2c16de931bebddc5d44d7b2.jpg  
      inflating: ./clothes_dataset/white_shoes/a94a76a556cc0e4ff0f4e5318e768d93e9de8234.jpg  
      inflating: ./clothes_dataset/white_shoes/a94ea572f12c3840eec213f58e28e68c8dbb68f6.jpg  
      inflating: ./clothes_dataset/white_shoes/a966329839dc3d4ed71014bed818a7913fe72032.jpg  
      inflating: ./clothes_dataset/white_shoes/a98fb633e66c4aff8a86d3c10275876e8dbc74bc.jpg  
      inflating: ./clothes_dataset/white_shoes/aa55b493e148bffbb89404988e3f04b1a02dd403.jpg  
      inflating: ./clothes_dataset/white_shoes/ab0ae99a9392af392393535bd5a210a50b836af9.jpg  
      inflating: ./clothes_dataset/white_shoes/ab1bca6fb659f502a3823b8b6083f5e088d83f85.jpg  
      inflating: ./clothes_dataset/white_shoes/ab7d5097ca73f9d052f33626b652bc9bb7b19cd7.jpg  
      inflating: ./clothes_dataset/white_shoes/abb15b2d9221cfa9f1a16fb4ac24b9d126916c93.jpg  
      inflating: ./clothes_dataset/white_shoes/abdccf8941705a44c8c51aff0a317a6b6859860f.jpg  
      inflating: ./clothes_dataset/white_shoes/acb483a95d9d9e8f6973866928f51e0536203818.jpg  
      inflating: ./clothes_dataset/white_shoes/acb8548f526d25ff80872bf8ac20b5560879dc64.jpg  
      inflating: ./clothes_dataset/white_shoes/ad5d2943edd0f689e285c90b6f228def9165fdac.jpg  
      inflating: ./clothes_dataset/white_shoes/ad6abb40d78c3778ed10c95e1012d5d6f5776012.jpg  
      inflating: ./clothes_dataset/white_shoes/addece455a8e62a52a7198484f2ff7ce536903b5.jpg  
      inflating: ./clothes_dataset/white_shoes/ae0aeec3ec1467535489ad614df1ce6676701aed.jpg  
      inflating: ./clothes_dataset/white_shoes/ae9c3caee23cf55c5c75e100a18ab206e34d756b.jpg  
      inflating: ./clothes_dataset/white_shoes/b14865eaab6d6a4c4fb8468edae14a5491c262ad.jpg  
      inflating: ./clothes_dataset/white_shoes/b1b1f400204d8e94632bda537cd5ad4ccf44fef7.jpg  
      inflating: ./clothes_dataset/white_shoes/b223e33cf28bbc566394683b5be3f7791bf2996d.jpg  
      inflating: ./clothes_dataset/white_shoes/b23a0d3a0db87dc8aee9b7a9c86339911a1d13c6.jpg  
      inflating: ./clothes_dataset/white_shoes/b2a779633cf05f604a1726b0039db609bbcc13c6.jpg  
      inflating: ./clothes_dataset/white_shoes/b2b51c6aa232a69dd9192ed729a178d91a83e0ff.jpg  
      inflating: ./clothes_dataset/white_shoes/b378e85b811cfef4ca6cf2104fbeabe2a685ae8e.jpg  
      inflating: ./clothes_dataset/white_shoes/b3f6bbaceb272561f10aef2965025751b6bf3196.jpg  
      inflating: ./clothes_dataset/white_shoes/b4c4bb29150551b441312f0f50ba11e49f1f0730.jpg  
      inflating: ./clothes_dataset/white_shoes/b53da7a63ad9c2599f0dc53ec879531c3de57962.jpg  
      inflating: ./clothes_dataset/white_shoes/b53e9797c1b6b35f049ee01a745f7e86411ec8b8.jpg  
      inflating: ./clothes_dataset/white_shoes/b54eb98ddefecf124ba13e48486d744756c4337b.jpg  
      inflating: ./clothes_dataset/white_shoes/b58c9aef4cdf8d2446534e5de634253663467127.jpg  
      inflating: ./clothes_dataset/white_shoes/b5ca05068a6f361dcfb682368c57d3fa5f0df973.jpg  
      inflating: ./clothes_dataset/white_shoes/b5e948a436eb89727106a6d42e780734f0bfd337.jpg  
      inflating: ./clothes_dataset/white_shoes/b60cfee28dea23ba356aac309ddc60f7612783a2.jpg  
      inflating: ./clothes_dataset/white_shoes/b66e71fadf626e8b491a8fec1580c4e86b2cfc9d.jpg  
      inflating: ./clothes_dataset/white_shoes/b6cc9133676c4bbc706901eee4a15e61e273e20a.jpg  
      inflating: ./clothes_dataset/white_shoes/b6f60ad72668f38fedc830b93d7428223d7a75ab.jpg  
      inflating: ./clothes_dataset/white_shoes/b7911a1f0195aecd16962b3b7696ae9508e11739.jpg  
      inflating: ./clothes_dataset/white_shoes/b7e2758e5a6ae4c31bdc5aa9ae99714b76aba5ad.jpg  
      inflating: ./clothes_dataset/white_shoes/b8a5ba2d9ccac3ab6b06bb9f01a0925a1d0d59dd.jpg  
      inflating: ./clothes_dataset/white_shoes/b9035fe45aa440f2e169cbb347e85975d0d15051.jpg  
      inflating: ./clothes_dataset/white_shoes/b9ccab89549dc82f92f17f58b16fab93670bb7bb.jpg  
      inflating: ./clothes_dataset/white_shoes/ba6ccbce710d285ce96a91885f45cc9432eb4bf1.jpg  
      inflating: ./clothes_dataset/white_shoes/bac1caf35085c99047f95304c17baa4712975112.jpg  
      inflating: ./clothes_dataset/white_shoes/bb4890b8f13dda2bfa20e72b500227c5f27f010a.jpg  
      inflating: ./clothes_dataset/white_shoes/bb5d34fb6214104ee5ccb4ef807d4b8700530955.jpg  
      inflating: ./clothes_dataset/white_shoes/bbf106d39422e42bfadd1563bc3a3c6153c6025e.jpg  
      inflating: ./clothes_dataset/white_shoes/bc20d87950598d6361e452eabc1b53d9f4f08234.jpg  
      inflating: ./clothes_dataset/white_shoes/bcd394c3d27f7eacc27f76d155f3873b28ba7c01.jpg  
      inflating: ./clothes_dataset/white_shoes/bcd9bdb6971280dc2667ea1f55092cd308bb50d2.jpg  
      inflating: ./clothes_dataset/white_shoes/bd585f9ce92b10bb114c284644295400968dd1b7.jpg  
      inflating: ./clothes_dataset/white_shoes/bdd3c5a08b79fcffc604174b7a8e28935dcfc284.jpg  
      inflating: ./clothes_dataset/white_shoes/bdd5aab3771f7be3a4e4c0060c5a77dacddece47.jpg  
      inflating: ./clothes_dataset/white_shoes/be7f3d2de096a44a46be452b865a69ab7f397505.jpg  
      inflating: ./clothes_dataset/white_shoes/beec4360736af6c0e932612a1a8a4bd6d3391c1c.jpg  
      inflating: ./clothes_dataset/white_shoes/bf0b74f6a653e28b5093875ee3ab004f3e6b3fb0.jpg  
      inflating: ./clothes_dataset/white_shoes/c08c0408f8f2794dd654ae851f0729dcd80b5b07.jpg  
      inflating: ./clothes_dataset/white_shoes/c11065da2e4fd9144948750efc6c942006bbf184.jpg  
      inflating: ./clothes_dataset/white_shoes/c17e7752800fffd3345dec0df093642d70151e5b.jpg  
      inflating: ./clothes_dataset/white_shoes/c22f9ce832c91530182507420f39aca81bb0b1eb.jpg  
      inflating: ./clothes_dataset/white_shoes/c2317f4f1bce6861959ea9d25fbaee1e5f02dfdb.jpg  
      inflating: ./clothes_dataset/white_shoes/c2b4b6f90631a077cd786ba020f49db5f3098520.jpg  
      inflating: ./clothes_dataset/white_shoes/c3d1d0e484ef455b396ee9189a8e0a16766161c4.jpg  
      inflating: ./clothes_dataset/white_shoes/c414736489f6d7d97f8db87180da41ee7eace031.jpg  
      inflating: ./clothes_dataset/white_shoes/c45f44167458a105cb0d7801c3d3332cbf743e4c.jpg  
      inflating: ./clothes_dataset/white_shoes/c4e3256591d868e1a5862a77b89b51ddbb3715a3.jpg  
      inflating: ./clothes_dataset/white_shoes/c548b4e4c788a6699169b81460865034edd7682d.jpg  
      inflating: ./clothes_dataset/white_shoes/c5ac97841e913fc3012e9ecc32160798b58392d6.jpg  
      inflating: ./clothes_dataset/white_shoes/c5ff07e728c67627574556df8b681abd07203f1e.jpg  
      inflating: ./clothes_dataset/white_shoes/c6990c20eba4364c45e8c3df482edc34bc019783.jpg  
      inflating: ./clothes_dataset/white_shoes/c756cd04928f1c9504f53612719fc3fd3af072ed.jpg  
      inflating: ./clothes_dataset/white_shoes/c83120d01c789280ae68bfd798ae0755b1f4bffe.jpg  
      inflating: ./clothes_dataset/white_shoes/c88143d509b2fca86995c5887ba87559a9d1e876.jpg  
      inflating: ./clothes_dataset/white_shoes/c891a5bf9dd11efefc5ada5fd4ee995fa517b941.jpg  
      inflating: ./clothes_dataset/white_shoes/ca8294d146227f8afcb017b513d8291cd12f8b82.jpg  
      inflating: ./clothes_dataset/white_shoes/cb56b7d346b89f82907394bd8e4602756959b68e.jpg  
      inflating: ./clothes_dataset/white_shoes/cc8979f3d9c72aa4fa0bea6cc2267fecd1a5481a.jpg  
      inflating: ./clothes_dataset/white_shoes/ccc3f6745ccb230e2ef8fd32149dbef9e30394fc.jpg  
      inflating: ./clothes_dataset/white_shoes/ccec68fb175e196397305ce28961b197602c2526.jpg  
      inflating: ./clothes_dataset/white_shoes/cd204b766c445519fb4865420f333b30165c38d8.jpg  
      inflating: ./clothes_dataset/white_shoes/ce7a6046133a8360a1381846c60efd81c8bc29e3.jpg  
      inflating: ./clothes_dataset/white_shoes/ceab4e64e2bccfb47d4c2e7000fc432e15248aee.jpg  
      inflating: ./clothes_dataset/white_shoes/cf1f6c9e3cc1ad22a9bbc1a179e5337abaa3a66a.jpg  
      inflating: ./clothes_dataset/white_shoes/cf23a30d2c59f73dbcbcb8210e7339a53d20b449.jpg  
      inflating: ./clothes_dataset/white_shoes/cfe7fc128ac6fb0392c24d9c68272e106f3786e7.jpg  
      inflating: ./clothes_dataset/white_shoes/d06bc6792e5b2c762a11d246eb690049d9fc8f2c.jpg  
      inflating: ./clothes_dataset/white_shoes/d0b500e175bdc0705c5a8746d4a1473f8fe957e7.jpg  
      inflating: ./clothes_dataset/white_shoes/d1307c04e87b555b0ae1b3dd91054fe4765929c2.jpg  
      inflating: ./clothes_dataset/white_shoes/d1ab979e21889d2f2553d3080202c868ea8f4c3c.jpg  
      inflating: ./clothes_dataset/white_shoes/d1bf05bc171101b40fc41762d9f8619e14803b34.jpg  
      inflating: ./clothes_dataset/white_shoes/d2a1257696114fe38bb76b2c120f8f98a5f9fb0e.jpg  
      inflating: ./clothes_dataset/white_shoes/d30089d0fcf0d2bb43c2eeab430b9c788c51ddde.jpg  
      inflating: ./clothes_dataset/white_shoes/d3180a25813a387b7b7bd0e39c54b3a9d6952708.jpg  
      inflating: ./clothes_dataset/white_shoes/d385870d62c7e56070a6812199e7deba6a317f5b.jpg  
      inflating: ./clothes_dataset/white_shoes/d38711110a7c1ab3371964b1af10abd05717b780.jpg  
      inflating: ./clothes_dataset/white_shoes/d388b6dbfedd9f318b5c6c30674a6a0e030f1e06.jpg  
      inflating: ./clothes_dataset/white_shoes/d3a7e6b66553f33afcbdfaa18cc23b42f66246a8.jpg  
      inflating: ./clothes_dataset/white_shoes/d3b1277cf62d7fef6707c09ab04e3335f8550073.jpg  
      inflating: ./clothes_dataset/white_shoes/d40bb6822c68c364f100fe34c09be4c152708868.jpg  
      inflating: ./clothes_dataset/white_shoes/d4a324cc2101baa2589be059c79ec072dd661ca1.jpg  
      inflating: ./clothes_dataset/white_shoes/d557f843d6714435403783ca2ea1553a0be57c29.jpg  
      inflating: ./clothes_dataset/white_shoes/d65563c34b7f62deb31a2ce772df9bf474b951c9.jpg  
      inflating: ./clothes_dataset/white_shoes/d75ab03c4fbdb44ddad98aa80a6131e14f83cf80.jpg  
      inflating: ./clothes_dataset/white_shoes/d77642269ef47dedf8bd3a501d0f1ca53d09143a.jpg  
      inflating: ./clothes_dataset/white_shoes/d7d113545b6a7f30ce21484c96f92eb8fd1a40ca.jpg  
      inflating: ./clothes_dataset/white_shoes/d8ab68d72e5061c9b8cc7d5f0d2654f9584ad350.jpg  
      inflating: ./clothes_dataset/white_shoes/d8dc4c7c969515924b36f3699a6e5bd195ff7ff5.jpg  
      inflating: ./clothes_dataset/white_shoes/d95c5845acd489e90415ba64d5468028173f985a.jpg  
      inflating: ./clothes_dataset/white_shoes/d95fff2e77a88e4732f2aae8555c00969edd08c8.jpg  
      inflating: ./clothes_dataset/white_shoes/d9607d168d0bfacae255ee0beeecbf61109e8dce.jpg  
      inflating: ./clothes_dataset/white_shoes/d9a421aa2b86f959625ee4501d99bc3328139753.jpg  
      inflating: ./clothes_dataset/white_shoes/d9c3aff3836393ee6d8af579584226b707f6c6e5.jpg  
      inflating: ./clothes_dataset/white_shoes/d9cfa0289d753be8c6258770d91cc24be4853e2b.jpg  
      inflating: ./clothes_dataset/white_shoes/da24b2bf32c81c40f20d48ff8899ced45a909740.jpg  
      inflating: ./clothes_dataset/white_shoes/db01466458c38185fadedd12c0b21d65262f7398.jpg  
      inflating: ./clothes_dataset/white_shoes/db313ced58a8ae89779dd8cecc603fce96906c28.jpg  
      inflating: ./clothes_dataset/white_shoes/db503fc58cdd486b498a0e02bab4238387c79959.jpg  
      inflating: ./clothes_dataset/white_shoes/dbcfea26e4d881f936e76e3f462b532e18a2e60f.jpg  
      inflating: ./clothes_dataset/white_shoes/dbe2177f9ec1c0be7b651c552e8dcf2cbd4cef93.jpg  
      inflating: ./clothes_dataset/white_shoes/dc1743727b6a75d4165c088b30130150df3ce917.jpg  
      inflating: ./clothes_dataset/white_shoes/dc208899d53d6816610e9d4517200f2c02d6ee52.jpg  
      inflating: ./clothes_dataset/white_shoes/dcabee26213f3449b42279be4e1e8b8b61c4ee5e.jpg  
      inflating: ./clothes_dataset/white_shoes/dd22dd947539f82910d5cb609fd529db17e2641c.jpg  
      inflating: ./clothes_dataset/white_shoes/dd3b0e103c7dd99db482c6cbd6439ed4a42f6634.jpg  
      inflating: ./clothes_dataset/white_shoes/dd5503bfd01e9c6260332fceecfc3900ac58d7c5.jpg  
      inflating: ./clothes_dataset/white_shoes/dd724d96080d06f305a7f99f499649aa0d927f3b.jpg  
      inflating: ./clothes_dataset/white_shoes/dde50a93b20992691f7790fe3e60615a2a5569c1.jpg  
      inflating: ./clothes_dataset/white_shoes/df1707ec18891968ce08dc96f80f53eb528283b3.jpg  
      inflating: ./clothes_dataset/white_shoes/dfded8982d299413dac3d113db207a87a48e91b3.jpg  
      inflating: ./clothes_dataset/white_shoes/e06b0e5408d18f3e9d06d6112c5b4a9a3b53fda4.jpg  
      inflating: ./clothes_dataset/white_shoes/e08285db01af05d7dd7dd647a981d3f9f775cd2a.jpg  
      inflating: ./clothes_dataset/white_shoes/e09a7346181cf6e25d9f279868e2b7d26703820b.jpg  
      inflating: ./clothes_dataset/white_shoes/e0a8ed68036dacb1b186b71fbf7219e899c24f1c.jpg  
      inflating: ./clothes_dataset/white_shoes/e1b12705339bd913224cd13ad831160afd4dc56c.jpg  
      inflating: ./clothes_dataset/white_shoes/e1d82a42cb0fe9c81245004e2ede81ef8a160f69.jpg  
      inflating: ./clothes_dataset/white_shoes/e20277e1c47391e651d6692c4b87da364e0c05a3.jpg  
      inflating: ./clothes_dataset/white_shoes/e31e7288504340ebde95503c141f17e42387e82a.jpg  
      inflating: ./clothes_dataset/white_shoes/e34155519fe4c5c07bdfb166ee129df5a0d97f94.jpg  
      inflating: ./clothes_dataset/white_shoes/e3fcc230b8df3747cb898fd299fe46dd8b2d0334.jpg  
      inflating: ./clothes_dataset/white_shoes/e415e6dafab1eb445bc6282370f57b2183ae9c76.jpg  
      inflating: ./clothes_dataset/white_shoes/e4631654011300f19d52988d0409462150b0dabf.jpg  
      inflating: ./clothes_dataset/white_shoes/e466e012a1856613175621a014e3ad5a578ea10c.jpg  
      inflating: ./clothes_dataset/white_shoes/e509879fceba2b69ce48c66633177cc16ce07d7d.jpg  
      inflating: ./clothes_dataset/white_shoes/e521439d12e511065931b6a457ef44108acd460c.jpg  
      inflating: ./clothes_dataset/white_shoes/e5b7b9ac2456d3760bda5f57c1e2fc4153037c4f.jpg  
      inflating: ./clothes_dataset/white_shoes/e5ef2df2577db1f870d47afe508807fcbfaaaaa7.jpg  
      inflating: ./clothes_dataset/white_shoes/e64cb3536bed7863a5d90a8fae710e7dab524885.jpg  
      inflating: ./clothes_dataset/white_shoes/e69b29c9479f36e7a23367b594e0133c3288b590.jpg  
      inflating: ./clothes_dataset/white_shoes/e6cc125864c9caa84a0c06c213445849f778fe52.jpg  
      inflating: ./clothes_dataset/white_shoes/e75ee134666408fd4b935b5572bdc84b5ab9b990.jpg  
      inflating: ./clothes_dataset/white_shoes/e764ae91848e88d359ce48ab9bf01047663a27b0.jpg  
      inflating: ./clothes_dataset/white_shoes/e7b81d876015ad0e23735a9e89f2857a9f4ef1e1.jpg  
      inflating: ./clothes_dataset/white_shoes/e8b4a5973de943eaba076ec2cbc35ead49f0e452.jpg  
      inflating: ./clothes_dataset/white_shoes/e9d2147d73436b30e83ceda9b40134e1720a3b4d.jpg  
      inflating: ./clothes_dataset/white_shoes/eb920e1ffc2805a33ca59db8916a48002e18ddda.jpg  
      inflating: ./clothes_dataset/white_shoes/eb99fb4584658ee598616c576423602f45549e8a.jpg  
      inflating: ./clothes_dataset/white_shoes/eb9f288d1e6c738e1119c2731376d371e84e23e3.jpg  
      inflating: ./clothes_dataset/white_shoes/ebcfaf9f8b700e72181393d2db0437f5d37be483.jpg  
      inflating: ./clothes_dataset/white_shoes/ec15d5183de012591f3b8c74dbb5841f64a68953.jpg  
      inflating: ./clothes_dataset/white_shoes/edd52e436efb1438b5d94ea7dcca21fea4c76de6.jpg  
      inflating: ./clothes_dataset/white_shoes/ee1f576f3c4957ef7b2d54995a43664746937354.jpg  
      inflating: ./clothes_dataset/white_shoes/ee2db26ea5b92ac27ffe68ad2cf54239d45be3fa.jpg  
      inflating: ./clothes_dataset/white_shoes/eee52eb74ad59e2894b5bb6253c6738a6ce2bff4.jpg  
      inflating: ./clothes_dataset/white_shoes/ef089f75c96aaa6a80e25c615eddfea688f55eb2.jpg  
      inflating: ./clothes_dataset/white_shoes/ef3346f8dbb0a40871e92722b6363eb4f1191e3e.jpg  
      inflating: ./clothes_dataset/white_shoes/efb5a553938da67487fffa4e022c3c8c22cdfcf9.jpg  
      inflating: ./clothes_dataset/white_shoes/f08a6b48ab18fd551381a9961b9ad01a15d12ce2.jpg  
      inflating: ./clothes_dataset/white_shoes/f100a56d829214e33d22fec5af1becfedfdf29ea.jpg  
      inflating: ./clothes_dataset/white_shoes/f10eb7ce170a63f38bdde980a2fa29505b8aa849.jpg  
      inflating: ./clothes_dataset/white_shoes/f2040331d4689731a4e8324119b143efce93d8a4.jpg  
      inflating: ./clothes_dataset/white_shoes/f2347440ccfe1372fb547392e8bfda18c7c96a90.jpg  
      inflating: ./clothes_dataset/white_shoes/f3361e34c1f92392a71a560f2a123487bc127692.jpg  
      inflating: ./clothes_dataset/white_shoes/f34922170e7daf3dbb116428e61d6b9d76569dca.jpg  
      inflating: ./clothes_dataset/white_shoes/f414918c81daf204961fdcf43b3f501201d321a5.jpg  
      inflating: ./clothes_dataset/white_shoes/f52785d5bf9eae5633981ff09202b8c5764fea03.jpg  
      inflating: ./clothes_dataset/white_shoes/f5671328cdf2b85f49d3e60abcd5b7df2034d707.jpg  
      inflating: ./clothes_dataset/white_shoes/f5867c171221ae98a80e2e3d6db607695ec63f4f.jpg  
      inflating: ./clothes_dataset/white_shoes/f59c9700a400b44eadf10a389941aa39e3130a6d.jpg  
      inflating: ./clothes_dataset/white_shoes/f5ba1db94c464be520a8291ea735a301c149c772.jpg  
      inflating: ./clothes_dataset/white_shoes/f5e46f585db699278b8848863c895f621bd63110.jpg  
      inflating: ./clothes_dataset/white_shoes/f61549cb6ec33866bd6b593790fa5f1ff9f09586.jpg  
      inflating: ./clothes_dataset/white_shoes/f62252888c9b9f9e70a5e6ecc07e107cae4c7c79.jpg  
      inflating: ./clothes_dataset/white_shoes/f68bcacefd7a81e6699894a23d2fde39781d7119.jpg  
      inflating: ./clothes_dataset/white_shoes/f6bce1dc67b0fff7897b0b6fae79c566827a92cc.jpg  
      inflating: ./clothes_dataset/white_shoes/f76930e402e20bffaff71f6e93049449fbf4cc47.jpg  
      inflating: ./clothes_dataset/white_shoes/f7c2a4719e9729737c86a07e66df1a4e0f60ee1e.jpg  
      inflating: ./clothes_dataset/white_shoes/f7efe38e0c3ccc9ac12fc3ca647130c5811f43ab.jpg  
      inflating: ./clothes_dataset/white_shoes/f8d461b67b3e4bcc0771d57955cbf07b1054987c.jpg  
      inflating: ./clothes_dataset/white_shoes/f8ed33ccbd8ef84c5b223c2d29fdc9f639760847.jpg  
      inflating: ./clothes_dataset/white_shoes/f9662c1b16db768eb722a8a952795b0c56d7368d.jpg  
      inflating: ./clothes_dataset/white_shoes/f9d658edce497c0a2d5d211fcf7cec6d5256a35d.jpg  
      inflating: ./clothes_dataset/white_shoes/f9d7841d887bc9d28663313ab7111c10eb4d882b.jpg  
      inflating: ./clothes_dataset/white_shoes/fa1453c98e0703cbe7db25226910ffbf821a1300.jpg  
      inflating: ./clothes_dataset/white_shoes/fa2c48d6caec04476e05bad0e634110290917bce.jpg  
      inflating: ./clothes_dataset/white_shoes/fa4ff0f0d7cec0d95d76b3ecc5b1b66b9312a937.jpg  
      inflating: ./clothes_dataset/white_shoes/faebb7c98b7b185e7093879f2afbe67758933b48.jpg  
      inflating: ./clothes_dataset/white_shoes/fb0fbbcc7f7b1b473ebe5682659f64d42d8f86cd.jpg  
      inflating: ./clothes_dataset/white_shoes/fbb66a831797154805ef38746dd4577613ca45f3.jpg  
      inflating: ./clothes_dataset/white_shoes/fbc11e253530bf70ce908d84be9d2e6576eadfc9.jpg  
      inflating: ./clothes_dataset/white_shoes/fc1bff9633494f0c76d46605ea81178307f6c108.jpg  
      inflating: ./clothes_dataset/white_shoes/fd68129c04d254f22994e8cee9eccac5c3bea864.jpg  
      inflating: ./clothes_dataset/white_shoes/fd9355e93c7122fca227186e2e5f2a725bda279e.jpg  
      inflating: ./clothes_dataset/white_shoes/fdb84354d5e9471bdfde5b3535d7c35519379d86.jpg  
      inflating: ./clothes_dataset/white_shoes/fe5229fbb118339759b8adf1ae4b3785465342e5.jpg  
      inflating: ./clothes_dataset/white_shoes/fe9664a7b5236a3d74b1bb0ea80f1db7e8ed7d60.jpg  
      inflating: ./clothes_dataset/white_shoes/fee0e462bdc81c769d80091d9cb2580a3ad5f05b.jpg  
      inflating: ./clothes_dataset/white_shoes/ff0b6b7caff9519d5010b88901e21ece70363522.jpg  
      inflating: ./clothes_dataset/white_shoes/ff36f33fc683f78caef7435fe758135c42e10c5b.jpg  
      inflating: ./clothes_dataset/white_shoes/ff3f0c285865662ac0ee2360384cbf861f077185.jpg  
      inflating: ./clothes_dataset/white_shoes/ff4007146a7a56ca8d1ce2f1203796e671bcd75a.jpg  
      inflating: ./clothes_dataset/white_shoes/fff93ca6b8ffe0888a5169a7ec84c66c9b9fe4b0.jpg  
      inflating: ./clothes_dataset/white_shorts/011a1213ff68595caed129353dd5cfe93a1e814f.jpg  
      inflating: ./clothes_dataset/white_shorts/03dc17892eef83c234d2dc7a0665c214e4711612.jpg  
      inflating: ./clothes_dataset/white_shorts/044f44f42afe6592a9bedaeaaacd480b298f7e53.jpg  
      inflating: ./clothes_dataset/white_shorts/0641516acfd4718d005194c8b8e9b9230b5ca289.jpg  
      inflating: ./clothes_dataset/white_shorts/0787127c9222afa9db0edfab1ca2a57d984992f8.jpg  
      inflating: ./clothes_dataset/white_shorts/09c25000860e4ecc8bf17eb1539f1705294ecd32.jpg  
      inflating: ./clothes_dataset/white_shorts/0c801d80ff53caae872f7b3df236ac778f47cd38.jpg  
      inflating: ./clothes_dataset/white_shorts/0cc9c09d616161a86bc69b3b471e8b2f9dca3aaa.jpg  
      inflating: ./clothes_dataset/white_shorts/0d9614e0f3a8235fe4fc9834a74f0106b1563a32.jpg  
      inflating: ./clothes_dataset/white_shorts/0edf4d1444b5f3cbfabb6724bad9ebc50cb96668.jpg  
      inflating: ./clothes_dataset/white_shorts/102c9127fead157dde4b1c3f2c73fd4566dc3fc2.jpg  
      inflating: ./clothes_dataset/white_shorts/10bb809caae47e847aafb6b17f16083bb5cada8d.jpg  
      inflating: ./clothes_dataset/white_shorts/13346e0df36d892bacdb14ea57e8ba4d6e73d486.jpg  
      inflating: ./clothes_dataset/white_shorts/1aa2f2818b7d56f0b7713fbdd6646748bd86a803.jpg  
      inflating: ./clothes_dataset/white_shorts/1edfce69371abb550cbf9b1710bebe7d3cfa495e.jpg  
      inflating: ./clothes_dataset/white_shorts/1ef07e256c5ff4ec2fa268762cde0a09912b9bf0.jpg  
      inflating: ./clothes_dataset/white_shorts/1f07602907266158d1537ec61d3e4180c1416c42.jpg  
      inflating: ./clothes_dataset/white_shorts/1fa4a411cf41f09dc39ed88b7d33e74ec2f8039e.jpg  
      inflating: ./clothes_dataset/white_shorts/213fb0c8abb4c4c48e0d910562956736aa8c7a85.jpg  
      inflating: ./clothes_dataset/white_shorts/21b8b118a91cdc9efcc3d4495f52fbfb1256403c.jpg  
      inflating: ./clothes_dataset/white_shorts/2244cc48358534757099d211b0edac726b8de945.jpg  
      inflating: ./clothes_dataset/white_shorts/288071361e12db95f6ca289f8c3836065ae9f0a4.jpg  
      inflating: ./clothes_dataset/white_shorts/2ace989900712e6dc688b47a85c97055362e09ac.jpg  
      inflating: ./clothes_dataset/white_shorts/3076c455364050a883fa0e9cc40f0c4385a062b7.jpg  
      inflating: ./clothes_dataset/white_shorts/37b0119b9c7e573faaa976cc2e3fdd9f5412fa08.jpg  
      inflating: ./clothes_dataset/white_shorts/3829d9a2cb237ebd0e2c7a86b587597c3f98ea50.jpg  
      inflating: ./clothes_dataset/white_shorts/3869a27a8507d06a89202778bae22b9a90141153.jpg  
      inflating: ./clothes_dataset/white_shorts/39777f99a971f7de4df9de4c8a2e395e4d5b7555.jpg  
      inflating: ./clothes_dataset/white_shorts/39a824e0bb92bb07ec412f86d783dccef8597112.jpg  
      inflating: ./clothes_dataset/white_shorts/3cd5bba1a2e3242915c530d64c113edbf4a6c2ce.jpg  
      inflating: ./clothes_dataset/white_shorts/3f9d396fe1e00f65c91c01fc2f1e52215a45a159.jpg  
      inflating: ./clothes_dataset/white_shorts/420ac7ee35c34b21d6f1b5f9b87b63d7bf09a8ed.jpg  
      inflating: ./clothes_dataset/white_shorts/44900ef5c31d4adae2f3cfe4f801a95391b8cfde.jpg  
      inflating: ./clothes_dataset/white_shorts/44f876472e6509c52bd423cac5af7bf3a7e334aa.jpg  
      inflating: ./clothes_dataset/white_shorts/457d34c42ae383c1e8f2eea43b0d2c452c0d4b18.jpg  
      inflating: ./clothes_dataset/white_shorts/45d954aa86358cc3a8a63895203ae2bc9aae4eed.jpg  
      inflating: ./clothes_dataset/white_shorts/47fdab73d0c637747a01013d36d17ada96b207c3.jpg  
      inflating: ./clothes_dataset/white_shorts/49f742d7856ae6c5709d39486cd6f22749c4d15b.jpg  
      inflating: ./clothes_dataset/white_shorts/49ff222dfe0600d8d401cf5bff95d1581a5b5eb3.jpg  
      inflating: ./clothes_dataset/white_shorts/4a3214c4cfb40fbc6d7391945a7ffc4bbc719a09.jpg  
      inflating: ./clothes_dataset/white_shorts/4abe0132cd942aa657264f148c485f6ddba79768.jpg  
      inflating: ./clothes_dataset/white_shorts/4cb2246b99ab367f0546d9708719b1d60c8e4662.jpg  
      inflating: ./clothes_dataset/white_shorts/50798776765c7ecf80af4ddae43d092b8386fd3f.jpg  
      inflating: ./clothes_dataset/white_shorts/5216b2cc0261577bac6bcf1b5b6a992bcc74e068.jpg  
      inflating: ./clothes_dataset/white_shorts/553f9e92d1dfbf1cd78e0472d21261bb87702822.jpg  
      inflating: ./clothes_dataset/white_shorts/57c8416e09eb2bf8d9576b16c0c50dff2a7dbe5a.jpg  
      inflating: ./clothes_dataset/white_shorts/5c1a8d66299fbfe7d9e4216210d91afa222ecd0a.jpg  
      inflating: ./clothes_dataset/white_shorts/5c4162f57538456dda71683fe8203ff8db763032.jpg  
      inflating: ./clothes_dataset/white_shorts/5dc5ac90f9d6c6baed9ee0968dd14d9ca9a99ccb.jpg  
      inflating: ./clothes_dataset/white_shorts/5f59963b2e7d4b831d28e4f768a90416a2bf3658.jpg  
      inflating: ./clothes_dataset/white_shorts/61dbe9a4ffcd95b1ea587541b1e473342cb68314.jpg  
      inflating: ./clothes_dataset/white_shorts/62be792264ca111e993f8a6ecf8d99bf3c3d938b.jpg  
      inflating: ./clothes_dataset/white_shorts/64924b74bb2812cf4f79521e103e6d73a1e43a3a.jpg  
      inflating: ./clothes_dataset/white_shorts/655fc8b60fffc44ae1c7c6e378b7bf1735ec7346.jpg  
      inflating: ./clothes_dataset/white_shorts/656aee1ac92de3e9f0bb541a503bfc2c43046a03.jpg  
      inflating: ./clothes_dataset/white_shorts/67779ffb53ef3a54b965634ae507bff4af59d26e.jpg  
      inflating: ./clothes_dataset/white_shorts/6a1bc7858fce94ca35f3c612810fca7312b4caad.jpg  
      inflating: ./clothes_dataset/white_shorts/6ac35404c84e184bd24ea1b85ddfda5cd3a70083.jpg  
      inflating: ./clothes_dataset/white_shorts/76e6dedaa758b0dedfc2aa83c067b279b1fe00b8.jpg  
      inflating: ./clothes_dataset/white_shorts/7729feda64a7c0c638dfc3b84d24700a57563457.jpg  
      inflating: ./clothes_dataset/white_shorts/77f3de0932938ece525cc79317a1d0fb849e9c51.jpg  
      inflating: ./clothes_dataset/white_shorts/783971c9d70fbee04fc1a85ca143d0864b3b24d9.jpg  
      inflating: ./clothes_dataset/white_shorts/7a86cbf3f780abdf1645143c3aeb0979f15b8437.jpg  
      inflating: ./clothes_dataset/white_shorts/7fdbc18deff6f7d5ad1f8c228061aa6bcecff535.jpg  
      inflating: ./clothes_dataset/white_shorts/847b6da6a74fce71af1096d228de644c3b39794a.jpg  
      inflating: ./clothes_dataset/white_shorts/85ab816dd21fb674106ef8bcdf753e043e9a6bc9.jpg  
      inflating: ./clothes_dataset/white_shorts/85e8b1e21f7e38d2bc068742cb5930c016bb65d4.jpg  
      inflating: ./clothes_dataset/white_shorts/88bc070003eb094f821a0e4a282593845b0b9a83.jpg  
      inflating: ./clothes_dataset/white_shorts/88fd4b08d106d6e995d8f986be7b347da95e40ab.jpg  
      inflating: ./clothes_dataset/white_shorts/8901e396a2907294880f24666212c519d9214759.jpg  
      inflating: ./clothes_dataset/white_shorts/8de895eb0c61e71b2cf7c958a8092de735d2d7b6.jpg  
      inflating: ./clothes_dataset/white_shorts/8f848aa8b5bf7ffbdcd00d7582149671afd74d87.jpg  
      inflating: ./clothes_dataset/white_shorts/90e2b0c10b8e19a93923c2dbcb7c3d0f6b29f9d7.jpg  
      inflating: ./clothes_dataset/white_shorts/9224f6b836409c2b592c2e2a728f81bbe57eaea6.jpg  
      inflating: ./clothes_dataset/white_shorts/950eee67acd622697a8a4a65ac81f303baa111bf.jpg  
      inflating: ./clothes_dataset/white_shorts/955c71fac2c3f0e4a7fcb31848690455d88d2907.jpg  
      inflating: ./clothes_dataset/white_shorts/956c12c980f3f3d576d84d584dfe814150645a2b.jpg  
      inflating: ./clothes_dataset/white_shorts/9ea9cb28a80400e255e0f7644ab28fed1b23a099.jpg  
      inflating: ./clothes_dataset/white_shorts/9fa70fdbeba8237980c470a948653b16bb65c87d.jpg  
      inflating: ./clothes_dataset/white_shorts/a07dcb2eee0aad9a3765e8692cedc143621660e3.jpg  
      inflating: ./clothes_dataset/white_shorts/a214ab3aab3f6b63e81f00bc55f702b1c0730dc9.jpg  
      inflating: ./clothes_dataset/white_shorts/a27938077e7b5cb1e45a6e9395a8dae819fd643a.jpg  
      inflating: ./clothes_dataset/white_shorts/a58a8d4e81f39313cd5cbad64bba21e7c9357964.jpg  
      inflating: ./clothes_dataset/white_shorts/ab2956d9cc1cf0337aa641a2e588348b44245528.jpg  
      inflating: ./clothes_dataset/white_shorts/aecb0af978d6676d716d6702b0f292481270d80e.jpg  
      inflating: ./clothes_dataset/white_shorts/af22731e9ce19af484a6d3d465693321c1de17f0.jpg  
      inflating: ./clothes_dataset/white_shorts/af3585e8f0904bf0acc06a60c43d1c48024fd49a.jpg  
      inflating: ./clothes_dataset/white_shorts/b0a6e36970dc4132ccef9706cdaf317161f5617b.jpg  
      inflating: ./clothes_dataset/white_shorts/b1288299b7396b46f55880386bf91ad6d585d468.jpg  
      inflating: ./clothes_dataset/white_shorts/b392fc5da884e95e77d5f4b1378e686949f9fc26.jpg  
      inflating: ./clothes_dataset/white_shorts/b4e4c808c2a198d3e84966300f45eede59b05cf8.jpg  
      inflating: ./clothes_dataset/white_shorts/b628698f58e2dd54e613053e8fd2e4db09c0c508.jpg  
      inflating: ./clothes_dataset/white_shorts/b8370088ff164f4b5a028ed1f5b571ce88f7a605.jpg  
      inflating: ./clothes_dataset/white_shorts/b88fef933553a0983dca29cbcb2eb02542b2037b.jpg  
      inflating: ./clothes_dataset/white_shorts/ba8db78fa15dd477c5daf2fbcd03eb8b086c3688.jpg  
      inflating: ./clothes_dataset/white_shorts/bbbc838d5c8557e6dbf6e5e2205f3594228f0308.jpg  
      inflating: ./clothes_dataset/white_shorts/be96dd228dc8c22ec39be79f334d9bb01363686f.jpg  
      inflating: ./clothes_dataset/white_shorts/bea1d40666568dc80c4ffede9cb1a6cee75a9c57.jpg  
      inflating: ./clothes_dataset/white_shorts/bef3a59f127aeed97afa1b445891c49c5b799333.jpg  
      inflating: ./clothes_dataset/white_shorts/c7ff3304a9fee3b38c788d4ee4721131b7c452e5.jpg  
      inflating: ./clothes_dataset/white_shorts/cb1025a7ce8a20b411fe5ca5d3623e274f5adf34.jpg  
      inflating: ./clothes_dataset/white_shorts/cda9ac1ad00971479ce133fc4459c32447fabd95.jpg  
      inflating: ./clothes_dataset/white_shorts/ce432b217e1140e89f58600db0e6f815a2034199.jpg  
      inflating: ./clothes_dataset/white_shorts/cf76cedee5d0735f19e89ad2e75f2bcba41c5991.jpg  
      inflating: ./clothes_dataset/white_shorts/d10d197c5b04ef97765a4f9464f42b48f601832c.jpg  
      inflating: ./clothes_dataset/white_shorts/d64aa04c96ab5bce3095baa05728a3e3a676d75f.jpg  
      inflating: ./clothes_dataset/white_shorts/d8fe819f56aee0ab5d4a6d78ce5ecf1c3adab35c.jpg  
      inflating: ./clothes_dataset/white_shorts/d9e7104765b62bff827e3c8824152e1b3b839255.jpg  
      inflating: ./clothes_dataset/white_shorts/da6da38ba3a08d9f65bc79431892d74b59186301.jpg  
      inflating: ./clothes_dataset/white_shorts/de36d13ca0474ad557c6dca012ce2b848f1d7467.jpg  
      inflating: ./clothes_dataset/white_shorts/e0bdea73dad23d41114bdf5e698ec3f048c7c6e5.jpg  
      inflating: ./clothes_dataset/white_shorts/e191e62ba495eb4fe62de84e71c40982ece39d05.jpg  
      inflating: ./clothes_dataset/white_shorts/ea421fad0abbb1fbf8e7d3d98fe7222ecfcc9117.jpg  
      inflating: ./clothes_dataset/white_shorts/eb0887d35ada8c331622fe3b7333d115a8c51777.jpg  
      inflating: ./clothes_dataset/white_shorts/ee1f52018378af4af32ba51425e6ef50381768b3.jpg  
      inflating: ./clothes_dataset/white_shorts/f0be9c6655655fe2166f2471dbc067c8f8692ff0.jpg  
      inflating: ./clothes_dataset/white_shorts/f35256887334f9f2f836435885f24ed28cf650bd.jpg  
      inflating: ./clothes_dataset/white_shorts/f50bf2c5e771f7392276a5e6b511fa7f053a1bbc.jpg  
      inflating: ./clothes_dataset/white_shorts/fb625925bc55cd5c0a147d8214b63d806ac7ecb1.jpg  
      inflating: ./clothes_dataset/white_shorts/fd76a8fe7e6c5d2fe5f346c3207f75325a781d82.jpg  
    

* 데이터 불러오기


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import glob as glob
import cv2

all_data = np.array(glob.glob('/content/clothes_dataset/*/*.jpg', recursive=True))

# 색과 옷의 종류를 구별하기 위해 해당되는 label에 1을 삽입합니다.
def check_cc(color, clothes):
    labels = np.zeros(11,)
    
    # color check
    if(color == 'black'):
        labels[0] = 1
        color_index = 0
    elif(color == 'blue'):
        labels[1] = 1
        color_index = 1
    elif(color == 'brown'):
        labels[2] = 1
        color_index = 2
    elif(color == 'green'):
        labels[3] = 1
        color_index = 3
    elif(color == 'red'):
        labels[4] = 1
        color_index = 4
    elif(color == 'white'):
        labels[5] = 1
        color_index = 5
        
    # clothes check
    if(clothes == 'dress'):
        labels[6] = 1
    elif(clothes == 'shirt'):
        labels[7] = 1
    elif(clothes == 'pants'):
        labels[8] = 1
    elif(clothes == 'shorts'):
        labels[9] = 1
    elif(clothes == 'shoes'):
        labels[10] = 1
        
    return labels, color_index

# label과 color_label을 담을 배열을 선언합니다.
all_labels = np.empty((all_data.shape[0], 11))
all_color_labels = np.empty((all_data.shape[0], 1))
# print(all_data[0])
for i, data in enumerate(all_data):
    color_and_clothes = all_data[i].split('/')[-2].split('_')

    color = color_and_clothes[0]
    clothes = color_and_clothes[1]
    # print(color,clothes)
    
    labels, color_index = check_cc(color, clothes)
    all_labels[i] = labels
    all_color_labels[i] = color_index
    
all_labels = np.concatenate((all_labels, all_color_labels), axis = -1)
```


```python
all_data
```




    array(['/content/clothes_dataset/red_shoes/57007b1e36f9b86f2832005bf20de8d3fe12b518.jpg',
           '/content/clothes_dataset/red_shoes/5e006b1eab73efeaa91fb76aa7c2d6e24706e60f.jpg',
           '/content/clothes_dataset/red_shoes/c23f9fcb3caebad169fd4b671cf71fd196fed7e3.jpg',
           ...,
           '/content/clothes_dataset/blue_shirt/83c86d0baf7782dc40aced68d451ad835bce930c.jpg',
           '/content/clothes_dataset/blue_shirt/7b0dae0a9bd09af24390c50089e14ed5874c060c.jpg',
           '/content/clothes_dataset/blue_shirt/c93ff1693d6d827ff4262c7bcf24c0d44ce397be.jpg'],
          dtype='<U82')




```python
all_labels.shape
```




    (11385, 12)




```python
from sklearn.model_selection import train_test_split

# 훈련, 검증, 테스트 데이터셋으로 나눕니다.
train_x, test_x, train_y, test_y = train_test_split(all_data, all_labels, shuffle = True, test_size = 0.3,
                                                   random_state = 99)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, shuffle = True, test_size = 0.3,
                                                 random_state = 99)
```


```python
train_df = pd.DataFrame({'image':train_x, 'black':train_y[:, 0], 'blue':train_y[:, 1],
                        'brown':train_y[:, 2], 'green':train_y[:, 3], 'red':train_y[:, 4],
                        'white':train_y[:, 5], 'dress':train_y[:, 6], 'shirt':train_y[:, 7],
                        'pants':train_y[:, 8], 'shorts':train_y[:, 9], 'shoes':train_y[:, 10],
                        'color':train_y[:, 11]})

val_df = pd.DataFrame({'image':val_x, 'black':val_y[:, 0], 'blue':val_y[:, 1],
                        'brown':val_y[:, 2], 'green':val_y[:, 3], 'red':val_y[:, 4],
                        'white':val_y[:, 5], 'dress':val_y[:, 6], 'shirt':val_y[:, 7],
                        'pants':val_y[:, 8], 'shorts':val_y[:, 9], 'shoes':val_y[:, 10],
                        'color':val_y[:, 11]})

test_df = pd.DataFrame({'image':test_x, 'black':test_y[:, 0], 'blue':test_y[:, 1],
                        'brown':test_y[:, 2], 'green':test_y[:, 3], 'red':test_y[:, 4],
                        'white':test_y[:, 5], 'dress':test_y[:, 6], 'shirt':test_y[:, 7],
                        'pants':test_y[:, 8], 'shorts':test_y[:, 9], 'shoes':test_y[:, 10],
                        'color':test_y[:, 11]})
```


```python
train_df.head()
```





  <div id="df-27f4c4cc-2edf-4f54-8e6d-1f2ac6a62a64">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>black</th>
      <th>blue</th>
      <th>brown</th>
      <th>green</th>
      <th>red</th>
      <th>white</th>
      <th>dress</th>
      <th>shirt</th>
      <th>pants</th>
      <th>shorts</th>
      <th>shoes</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/content/clothes_dataset/green_shorts/e74d11d3...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/content/clothes_dataset/black_dress/f1be32393...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/content/clothes_dataset/black_shoes/04f78f68a...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/content/clothes_dataset/brown_pants/0671d132b...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/content/clothes_dataset/white_shoes/59803fb01...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-27f4c4cc-2edf-4f54-8e6d-1f2ac6a62a64')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-27f4c4cc-2edf-4f54-8e6d-1f2ac6a62a64 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-27f4c4cc-2edf-4f54-8e6d-1f2ac6a62a64');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 저장할 경로
!mkdir csv_data
```

    mkdir: cannot create directory ‘csv_data’: File exists
    


```python
# 저장
train_df.to_csv('/content/csv_data/train.csv', index=False)
val_df.to_csv('/content/csv_data/val.csv', index=False)
test_df.to_csv('/content/csv_data/test.csv', index=False)
```


```python
# 이미지 제너레이터 정의하기
from keras.preprocessing.image import ImageDataGenerator
```


```python
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

def get_steps(num_sampels,batch_size):
  if (num_sampels % batch_size) > 0 :
    return (num_sampels // batch_size) + 1
  else:
     return (num_sampels // batch_size)
```


```python
# 모델 만들기
from keras.models import Sequential
from keras.layers import Dense, Flatten
```


```python
model = Sequential()
model.add(Flatten(input_shape=(112,112,3))) # RGB값으로 색 지정
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(11,activation='sigmoid')) # 11개의 출력을 가지는 신경망
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 37632)             0         
                                                                     
     dense_3 (Dense)             (None, 128)               4817024   
                                                                     
     dense_4 (Dense)             (None, 64)                8256      
                                                                     
     dense_5 (Dense)             (None, 11)                715       
                                                                     
    =================================================================
    Total params: 4,825,995
    Trainable params: 4,825,995
    Non-trainable params: 0
    _________________________________________________________________
    


```python
train_df.columns
```




    Index(['image', 'black', 'blue', 'brown', 'green', 'red', 'white', 'dress',
           'shirt', 'pants', 'shorts', 'shoes', 'color'],
          dtype='object')




```python
# 데이터 제너레이터 정의하기
batch_size = 32
class_col =['black', 'blue', 'brown', 'green', 'red', 'white', 'dress','shirt', 'pants', 'shorts', 'shoes']

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    x_col='image',
                                                    y_col=class_col,
                                                    target_size=(112,112),
                                                    color_mode='rgb',
                                                    class_mode='raw',
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=42)
val_generator = val_datagen.flow_from_dataframe(dataframe=val_df,
                                                    x_col='image',
                                                    y_col=class_col,
                                                    target_size=(112,112),
                                                    color_mode='rgb',
                                                    class_mode='raw',
                                                    batch_size=batch_size,
                                                    shuffle=True
                                                 )
```

    Found 5578 validated image filenames.
    Found 2391 validated image filenames.
    


```python
model.fit(train_generator,
          steps_per_epoch=get_steps(len(train_df),batch_size), 
          validation_data=val_generator,
          validation_steps=get_steps(len(val_df),batch_size),
          epochs = 10)
```

    Epoch 1/10
    175/175 [==============================] - 28s 157ms/step - loss: 0.5666 - binary_accuracy: 0.8415 - val_loss: 0.2993 - val_binary_accuracy: 0.8875
    Epoch 2/10
    175/175 [==============================] - 29s 165ms/step - loss: 0.3014 - binary_accuracy: 0.8808 - val_loss: 0.3139 - val_binary_accuracy: 0.8778
    Epoch 3/10
    175/175 [==============================] - 27s 154ms/step - loss: 0.2451 - binary_accuracy: 0.9035 - val_loss: 0.2435 - val_binary_accuracy: 0.9037
    Epoch 4/10
    175/175 [==============================] - 34s 192ms/step - loss: 0.2215 - binary_accuracy: 0.9130 - val_loss: 0.2104 - val_binary_accuracy: 0.9182
    Epoch 5/10
    175/175 [==============================] - 32s 180ms/step - loss: 0.2134 - binary_accuracy: 0.9158 - val_loss: 0.2368 - val_binary_accuracy: 0.9118
    Epoch 6/10
    175/175 [==============================] - 30s 169ms/step - loss: 0.1952 - binary_accuracy: 0.9232 - val_loss: 0.2123 - val_binary_accuracy: 0.9210
    Epoch 7/10
    175/175 [==============================] - 28s 160ms/step - loss: 0.1900 - binary_accuracy: 0.9256 - val_loss: 0.1868 - val_binary_accuracy: 0.9272
    Epoch 8/10
    175/175 [==============================] - 33s 189ms/step - loss: 0.1800 - binary_accuracy: 0.9297 - val_loss: 0.2549 - val_binary_accuracy: 0.9053
    Epoch 9/10
    175/175 [==============================] - 31s 177ms/step - loss: 0.1699 - binary_accuracy: 0.9329 - val_loss: 0.1640 - val_binary_accuracy: 0.9369
    Epoch 10/10
    175/175 [==============================] - 32s 186ms/step - loss: 0.1587 - binary_accuracy: 0.9381 - val_loss: 0.2165 - val_binary_accuracy: 0.9186
    




    <keras.callbacks.History at 0x7f69924b0e20>




```python
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                    x_col='image',
                                                    target_size=(112,112),
                                                    color_mode='rgb',
                                                    class_mode=None,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                  )
preds = model.predict(test_generator,steps=get_steps(len(test_df),batch_size),verbose=1)
```

    Found 3416 validated image filenames.
    107/107 [==============================] - 12s 109ms/step
    


```python
np.round(preds[0],2)
```




    array([0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.02, 0.  ],
          dtype=float32)




```python
import matplotlib.pyplot as plt
```


```python
# 테스트 데이터 예측하기
do_preds = preds[:8]
for i ,pred in enumerate(do_preds):
  plt.subplot(2,4,i+1)
  prob = zip(class_col,list(pred))
  # print(list(prob))
  prob = sorted(list(prob),key=lambda x:x[1],reverse=True)
  # print(prob)
  image = cv2.imread(test_df['image'][i])
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.title(f'{prob[0][0]}:{round(prob[0][1]*100,2)}% \n{prob[1][0]}:{round(prob[1][1]*100,2)}%')
plt.tight_layout()
plt.show()
```


    
![png](06_clothes_files/06_clothes_25_0.png)
    



```python
data_datagen = ImageDataGenerator(rescale=1./255)

data_generator = data_datagen.flow_from_directory(
                                                    directory='/content/drive/MyDrive/Colab Notebooks/sesac_deeplearning/06_clothes_img',
                                                    target_size=(112,112),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                  )
result = model.predict(data_generator,steps=get_steps(2,batch_size),verbose=1)
```

    Found 4 images belonging to 1 classes.
    1/1 [==============================] - 0s 89ms/step
    


```python
result
```




    array([[1.52786851e-01, 6.52106421e-04, 9.56296504e-01, 1.85247824e-01,
            8.73344397e-05, 2.28309003e-03, 1.27795630e-03, 1.63213081e-05,
            4.72517684e-03, 2.22380459e-03, 9.80769515e-01],
           [4.02874887e-01, 1.39958924e-04, 2.90269911e-01, 2.09356956e-02,
            4.10276145e-04, 2.19159517e-02, 5.00542298e-03, 1.35187315e-06,
            1.68185332e-03, 1.23237018e-02, 9.77578521e-01],
           [1.79782207e-03, 4.62408469e-04, 4.34684247e-01, 6.77505648e-03,
            7.25663122e-05, 6.68699384e-01, 1.03075884e-01, 1.37986478e-06,
            2.92289741e-02, 2.66093817e-02, 7.58549571e-01],
           [9.93593596e-03, 6.96583709e-04, 1.71822265e-01, 2.91292294e-04,
            1.42630748e-03, 3.08194607e-01, 6.62644506e-02, 4.05802979e-07,
            1.50276301e-02, 2.48940680e-02, 6.62990630e-01]], dtype=float32)




```python
np.round(result,2)
```




    array([[0.15, 0.  , 0.96, 0.19, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.98],
           [0.4 , 0.  , 0.29, 0.02, 0.  , 0.02, 0.01, 0.  , 0.  , 0.01, 0.98],
           [0.  , 0.  , 0.43, 0.01, 0.  , 0.67, 0.1 , 0.  , 0.03, 0.03, 0.76],
           [0.01, 0.  , 0.17, 0.  , 0.  , 0.31, 0.07, 0.  , 0.02, 0.02, 0.66]],
          dtype=float32)




```python
for i, pred in enumerate(result):
  prob = zip(class_col,list(pred))
  prob = sorted(list(prob),key=lambda x:x[1],reverse=True)
  print((f'{prob[0][0]}:{round(prob[0][1]*100,2)}% \n{prob[1][0]}:{round(prob[1][1]*100,2)}%'))
```

    shoes:98.08% 
    brown:95.63%
    shoes:97.76% 
    black:40.29%
    shoes:75.85% 
    white:66.87%
    shoes:66.3% 
    white:30.82%
    


```python

```
