class ViT(nn.Module):
    def __init__(self, num_labels=2, model_name="google/vit-base-patch16-224"):
        super(ViT, self).__init__()
        
        self.vit = AutoModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size
        self.trilinear=nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, pixel_values, labels=None, skip_classifier=False):
        
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  
        pooled_output = self.trilinear(pooled_output)
        if skip_classifier:
            return pooled_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return (loss, logits) if loss is not None else logits

