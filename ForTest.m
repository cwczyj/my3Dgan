rng('shuffle');

load tmp_loss_file.mat
load tmp_generator.mat
load tmp_discriminator.mat

fprintf('begin test the procedure of bp in the discriminator net\n');
discriminator.weight_decay = 1e-5;
discriminator = myDiscriminator(discriminator,disc_Loss','backward','true');