rng('shuffle');

load tmp_generator.mat
load tmp_discriminator.mat

disc_Loss = log10(disc_output_real) + log10(1-disc_output_G);
disc_Loss = mean(disc_Loss(:));
disc_Loss = disc_Loss*ones(10,1);
discriminator = myDiscriminator(discriminator,disc_Loss','backward','true');

if 0
rand_z = rand([200,10],'single');
generator = myGenerator(generator,rand_z,'forward');
gen_output_2 = generator.layers{6}.input{1};
    
discriminator = myDiscriminator(discriminator,gen_output_2,'forward','true');
    
disc_output_G = discriminator.layers{6}.input{1};
    
disc_Loss = log10(1-disc_output_G);
disc_Loss = mean(disc_Loss(:));
disc_Loss = disc_Loss*ones(10,1);
    
discriminator = myDiscriminator(discriminator,disc_Loss','backward','false');
gen_Loss = discriminator.layers{1}.dinput{1};
  


generator = myGenerator(generator,gen_Loss,'backward');
end