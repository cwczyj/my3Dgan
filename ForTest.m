rng('shuffle');

load tmp_loss_file.mat
load tmp_generator.mat
load tmp_discriminator.mat

fprintf('begin test the procedure of bp in the discriminator net\n');
discriminator = myDiscriminator(discriminator,disc_Loss','backward','true');