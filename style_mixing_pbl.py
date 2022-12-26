# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
import random

#----------------------------------------------------------------------------
# 모델을 url로부터 불러오는 코드

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()

def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


#----------------------------------------------------------------------------
# Style mixing.

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    print("src_latents:",src_latents,src_latents.shape, type(src_latents))
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    print("src_dlatents:",src_dlatents, src_dlatents.shape, type(src_dlatents))
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    print("dst_dlatents:",dst_dlatents, dst_dlatents.shape, type(dst_dlatents))
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

##noise True
def draw_style_mixing_figure_noise_True(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    print("src_latents:",src_latents,src_latents.shape, type(src_latents))
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    print("src_dlatents:",src_dlatents, src_dlatents.shape, type(src_dlatents))
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    print("dst_dlatents:",dst_dlatents, dst_dlatents.shape, type(dst_dlatents))
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=True, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=True, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=True, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

def style_mixing_figure_from_two_networks(png, Gs1, Gs2, w, h, src_seeds, dst_seeds, style_ranges):
    #이 함수의 경우, 합성된 사진에서 src_dlatents(Seed를 통해 나온 latent vector에 대해 Gs2의 Mapping 네트워크를 거친 w벡터)를
    #style_ranges에서 지정한 범위에 dst_dlatents(Seed를 통해 나온 latent vector에 대해 Gs1의 Mapping 네트워크를 거친 w벡터)를
    # 대체하여 집어 넣은 후, Gs2의 Synthesis 네트워크를 거쳐 이미지를 생성합니다.
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs2.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs1.input_shape[1]) for seed in dst_seeds)
    print(src_latents,"src_latents:",src_latents.shape, type(src_latents))
    src_dlatents = Gs1.components.mapping.run(src_latents, None) # [seed, layer, component]
    print(src_dlatents,"src_dlatents:", src_dlatents.shape, type(src_dlatents))
    dst_dlatents = Gs1.components.mapping.run(dst_latents, None) # [seed, layer, component]
    print(dst_dlatents,"dst_dlatents:", dst_dlatents.shape, type(dst_dlatents))
    src_images = Gs2.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs1.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    print("src_dlatents.shape[1]",src_dlatents.shape[1])
    if src_dlatents.shape[1]<dst_dlatents.shape[1]:  #src latent의 크기가 dst latent의 크기보다 작다면
        print("합성의 크기가 달라 조정합니다...")
        print("src_dlatents.shape:",src_dlatents.shape)
        numadd = dst_dlatents.shape[1]-src_dlatents.shape[1] #더 해줄 항의 개수
        src_dlatents_add=src_dlatents[:,0:numadd,:]
        print("src_dlatents_add.shape:",src_dlatents_add.shape)
        print("조정 중..")
        np.vstack([src_dlatents,src_dlatents_add])
        print("src_dlatents.shape:",src_dlatents.shpae)
        print("조정 완료")

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        print("row_dlatents.shape:",row_dlatents.shape,"src_dlatents.shape:",src_dlatents.shape,"dst_dlatents.shape:",dst_dlatents.shape)
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        print("style_ranges[row]:",style_ranges[row], "row:",row)
        row_images = Gs1.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    
    with tf.device("/gpu:0"): #어떤 GPU 사용할 지 선택      
        #모델 불러오기
        # model_name1='karras2019stylegan-ffhq-1024x1024.pkl'
        # model_name1='karras2019stylegan-celebahq-1024x1024.pkl'
        # model_name1='karras2019stylegan-cats-256x256.pkl'
        model_name1='karras2019stylegan-cars-512x384.pkl'
        # model_name1='black-network-snapshot-005726.pkl'
        # model_name1='child-network-snapshot-004705.pkl'
        # model_name1='ffhq_network-snapshot-010526.pkl'
        # model_name1='ffhq-network-snapshot-005306.pkl'

        # model_name2='karras2019stylegan-ffhq-1024x1024.pkl'
        # model_name2='karras2019stylegan-celebahq-1024x1024.pkl'
        model_name2='karras2019stylegan-cats-256x256.pkl'
        # model_name2='karras2019stylegan-cars-512x384.pkl'
        # model_name2='black-network-snapshot-005726.pkl'
        # model_name2='child-network-snapshot-004705.pkl'
        # model_name2='ffhq_network-snapshot-010526.pkl'
        # model_name2='ffhq-network-snapshot-005306.pkl'

        with open(model_name1,'rb') as f1:
            _G1, _D1, Gs = pickle.load(f1)

        with open(model_name2,'rb') as f2:
            _G2, _D2, Gs2 = pickle.load(f2)
        
        imgnum = 6 #이미지 개수 지정
        imgwidth=512 #생성 이미지 해상도
        imgheight=384 #생성 이미지 해상도
        seed1=random.randrange(1,10000) #1~10000 사이의 무작위 시드 생성
        seed2=random.randrange(1,10000) #1~10000 사이의 무작위 시드 생성
        # seed1 = 7557 #시드 직접 지정
        # seed2 = 3970 #시드 직접 지정
        seed1ar=[]
        seed2ar=[]
        two_network_seed1ar=[]
        two_network_seed2ar=[]
        for i in range(imgnum+1):
            two_network_seed1ar.append(seed1)
            two_network_seed2ar.append(seed2)
            if i==imgnum:continue
            seed1ar.append(seed1)
            seed2ar.append(seed2)

        #src vector가 dst vector에 들어가는 위치, style_ranges의 개수는 imgnum과 일치해야 함. 
        # style_ranges=[range(0,4)]+[range(4,8)]+[range(8,14)]+[range(4,11)]+[range(0,14,2)]+[range(1,14,2)] #사용되는 합성 네트워크의 해상도: 256x256
        style_ranges=[range(0,4)]+[range(4,8)]+[range(8,16)]+[range(4,12)]+[range(0,16,2)]+[range(1,16,2)] #사용되는 합성 네트워크의 해상도: 512x512, 512x384
        # style_ranges=[range(0,4)]+[range(4,8)]+[range(8,18)]+[range(4,13)]+[range(0,18,2)]+[range(1,18,2)] #사용되는 합성 네트워크의 해상도: 1024x1024
        
        #mixing_style_ranges는 Source Vector 자기 자신의 합성 네트워크를 지난 이미지를 표시해야하므로 이미지 개수 + 1
        mixing_style_ranges=[range(0,14)]+style_ranges #Source Network의 해상도 256x256
        #mixing_style_ranges=[range(0,16)]+style_ranges  #Source Network의 해상도 512x512, 512x384
        # mixing_style_ranges=[range(0,18)]+style_ranges #Source Network의 해상도 1024x1024
        
        direc='/'+model_name1[:-4]+model_name2[:-4]+'/seed1_'+str(seed1)+'seed2_'+str(seed2)+'/'
        os.makedirs(config.result_dir+direc, exist_ok=True)
        #어떤 vector 조합을 사용했는지 자동 작성
        rangetxt=open(config.result_dir+direc+'range.txt', 'w')
        rangetxt.write('style_ranges:'+str(style_ranges)+'mixing_style_ranges:'+str(mixing_style_ranges))
        rangetxt.close()


        draw_style_mixing_figure(os.path.join(config.result_dir+direc,'style-mixing_stylerangesrc'+str(seed1)+'dst'+str(seed2)+'.png'), Gs, w=imgwidth, h=imgheight, src_seeds=seed1ar, dst_seeds=seed2ar, style_ranges=style_ranges)
        # draw_style_mixing_figure_noise_True(os.path.join(config.result_dir+direc, 'style-mixing_stylerange_src'+str(seed1)+'dst'+str(seed2)+'.png'), Gs, w=imgwidth, h=imgwidth, src_seeds=seed1ar, dst_seeds=seed2ar, style_ranges=style_ranges)
        draw_style_mixing_figure(os.path.join(config.result_dir+direc, 'style-mixing_stylerangesrc'+str(seed2)+'dst'+str(seed1)+'.png'), Gs, w=imgwidth, h=imgheight, src_seeds=seed2ar, dst_seeds=seed1ar, style_ranges=style_ranges)
        # draw_style_mixing_figure_noise_True(os.path.join(config.result_dir+direc, 'style-mixing_stylerange_src'+str(seed2)+'dst'+str(seed1)+'.png'), Gs, w=imgwidth, h=imgwidth, src_seeds=seed2ar, dst_seeds=seed1ar, style_ranges=style_ranges)
        
        try:
            style_mixing_figure_from_two_networks(os.path.join(config.result_dir+direc, 'style-mixing-two_networks_stylerange_src'+str(seed1)+'dst'+str(seed2)+'.png'), Gs,Gs2, w=imgwidth, h=imgheight, src_seeds=two_network_seed1ar, dst_seeds=two_network_seed2ar, style_ranges=mixing_style_ranges)
        except:
            print("합성 네트워크의 해상도와 맞지 않아 합성이 불가능합니다.")
        try:
            style_mixing_figure_from_two_networks(os.path.join(config.result_dir+direc, 'style-mixing-two_networks_stylerange_src'+str(seed2)+'dst'+str(seed1)+'.png'), Gs,Gs2, w=imgwidth, h=imgheight, src_seeds=two_network_seed2ar, dst_seeds=two_network_seed1ar, style_ranges=mixing_style_ranges)
        except:
            print("합성 네트워크의 해상도와 맞지 않아 합성이 불가능합니다.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
