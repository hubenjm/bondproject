from PIL import Image
import tesserocr

pad = 7
f = Image.open('cusipimage.png')
old_size = f.size
new_size = (old_size[0] + 2*pad, old_size[1] + 2*pad)
new_im = Image.new("RGB", new_size, color = 'white')
new_im.paste(f, (pad, pad))
#pixels = new_im.load()

#for i in range(pad):
#    for j in range(pad, new_size[1]-pad): #left and right bands
#        pixels[i,j] = (255,255,255)
#        pixels[new_size[0]-1-i,j] = (255,255,255)

#for i in xrange(new_size[0]):
#    for j in xrange(pad): #top and bottom bands
#        pixels[i,j] = (255,255,255)
#        pixels[i,new_size[1]-1-j] = (255,255,255)

print(tesserocr.image_to_text(new_im)[:9])

