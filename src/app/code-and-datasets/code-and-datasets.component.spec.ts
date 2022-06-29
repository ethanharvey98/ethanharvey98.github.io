import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CodeAndDatasetsComponent } from './code-and-datasets.component';

describe('CodeAndDatasetsComponent', () => {
  let component: CodeAndDatasetsComponent;
  let fixture: ComponentFixture<CodeAndDatasetsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CodeAndDatasetsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CodeAndDatasetsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
